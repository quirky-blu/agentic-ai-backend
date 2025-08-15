from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import sqlite3
import psycopg2
from sqlalchemy import create_engine
import io
import json
import asyncio
from typing import Optional, Dict, Any, List
import uvicorn
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import re

# Azure AI Integration
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from duckduckgo_search import DDGS

# Load environment variables
load_dotenv()
token = os.getenv("GITHUB_MODELS_KEY")

# Azure AI client setup
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"
ai_client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token)
)

# FastAPI app setup
app = FastAPI(title="Data Analysis API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class DatabaseConnection(BaseModel):
    connection_string: str
    query: Optional[str] = None

class NaturalLanguageQuery(BaseModel):
    query: str
    data_source: str  # 'csv' or 'db'
    file_id: Optional[str] = None
    connection_string: Optional[str] = None

class AnalysisResult(BaseModel):
    query: str
    interpretation: str
    data: List[Dict[Any, Any]]
    visualization_type: str
    chart_config: Optional[Dict] = None
    sql_query: Optional[str] = None

# In-memory storage for uploaded files
uploaded_files = {}

# Database connection helper
def get_db_connection(connection_string: str):
    """Create database connection based on connection string"""
    try:
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Database connection failed: {str(e)}")

# AI-powered query analyzer
class AIDataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.column_info = self._get_column_info()
    
    def _get_column_info(self) -> str:
        """Get detailed information about dataset columns"""
        info = []
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            unique_count = self.df[col].nunique()
            null_count = self.df[col].isnull().sum()
            
            col_info = f"- {col} ({dtype}): {unique_count} unique values, {null_count} nulls"
            
            if self.df[col].dtype in ['int64', 'float64']:
                col_info += f", range: {self.df[col].min():.2f} to {self.df[col].max():.2f}"
            elif unique_count <= 10:
                sample_values = self.df[col].dropna().unique()[:5]
                col_info += f", sample values: {list(sample_values)}"
            
            info.append(col_info)
        
        return "\n".join(info)
    
    async def analyze_query(self, query: str) -> Dict:
        """Use AI to analyze natural language query and return results"""
        try:
            # Create system prompt with dataset context
            system_prompt = f"""You are a data analyst assistant. Analyze the user's natural language query and provide structured analysis.

Dataset Information:
- Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns
- Columns:
{self.column_info}

Your task:
1. Interpret the user's query
2. Determine the best analysis approach
3. Generate appropriate pandas operations
4. Suggest visualization type
5. Provide interpretation

Return your response as JSON with these keys:
- "interpretation": Clear explanation of what analysis will be performed
- "analysis_type": One of ["summary", "filter", "group", "top_values", "correlation", "trend", "comparison"]
- "pandas_code": Python code using 'df' variable to perform the analysis
- "visualization_type": One of ["table", "bar_chart", "line_chart", "scatter_plot", "histogram", "pie_chart"]
- "chart_config": Object with chart configuration (x, y, title, etc.)

Example query: "Show me top 5 customers by revenue"
Example response: {{"interpretation": "Finding the 5 customers with highest revenue", "analysis_type": "top_values", "pandas_code": "df.nlargest(5, 'revenue')[['customer_name', 'revenue']]", "visualization_type": "bar_chart", "chart_config": {{"x": "customer_name", "y": "revenue", "title": "Top 5 Customers by Revenue"}}}}
"""

            # Get AI response
            response = ai_client.complete(
                messages=[
                    SystemMessage(content=system_prompt),
                    UserMessage(content=f"User query: {query}")
                ],
                model=model,
                max_tokens=1000,
                temperature=0.3
            )
            
            # Parse AI response
            ai_response = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                analysis_plan = json.loads(json_match.group())
            else:
                # Fallback to simple analysis
                return self._fallback_analysis(query)
            
            # Execute the pandas code safely
            try:
                # Create safe environment for code execution
                safe_globals = {
                    'df': self.df,
                    'pd': pd,
                    'np': __import__('numpy')
                }
                
                result_df = eval(analysis_plan['pandas_code'], safe_globals)
                if isinstance(result_df, pd.Series):
                    result_df = result_df.to_frame()
                
                analysis_data = result_df.to_dict('records')
                
                return {
                    'interpretation': analysis_plan['interpretation'],
                    'data': analysis_data,
                    'visualization_type': analysis_plan['visualization_type'],
                    'chart_config': analysis_plan.get('chart_config'),
                    'analysis_type': analysis_plan.get('analysis_type', 'unknown'),
                    'pandas_code': analysis_plan['pandas_code']
                }
                
            except Exception as code_error:
                print(f"Code execution error: {code_error}")
                return self._fallback_analysis(query)
                
        except Exception as e:
            print(f"AI analysis error: {e}")
            return self._fallback_analysis(query)
    
    def _fallback_analysis(self, query: str) -> Dict:
        """Fallback analysis when AI fails"""
        query_lower = query.lower()
        
        if 'top' in query_lower and any(x in query_lower for x in ['sales', 'revenue', 'amount']):
            return self._get_top_values(query)
        elif 'group' in query_lower:
            return self._group_analysis(query)
        elif 'chart' in query_lower or 'plot' in query_lower:
            return self._create_visualization(query)
        elif 'filter' in query_lower:
            return self._filter_data(query)
        else:
            return self._general_summary(query)
    
    def _get_top_values(self, query: str) -> Dict:
        """Get top N values from dataset"""
        # Extract number if present
        numbers = re.findall(r'\d+', query)
        n = int(numbers[0]) if numbers else 10
        
        # Find numeric column
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            top_data = self.df.nlargest(n, col).to_dict('records')
            
            return {
                'interpretation': f'Showing top {n} records by {col}',
                'data': top_data,
                'visualization_type': 'bar_chart',
                'chart_config': {
                    'x': self.df.columns[0],
                    'y': col,
                    'title': f'Top {n} by {col}'
                }
            }
        
        return self._general_summary(query)
    
    def _group_analysis(self, query: str) -> Dict:
        """Group data analysis"""
        # Simple grouping by first categorical column
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            group_col = categorical_cols[0]
            value_col = numeric_cols[0]
            
            grouped = self.df.groupby(group_col)[value_col].agg(['sum', 'mean', 'count']).reset_index()
            grouped_data = grouped.to_dict('records')
            
            return {
                'interpretation': f'Data grouped by {group_col} with {value_col} statistics',
                'data': grouped_data,
                'visualization_type': 'bar_chart',
                'chart_config': {
                    'x': group_col,
                    'y': 'sum',
                    'title': f'{value_col} by {group_col}'
                }
            }
        
        return self._general_summary(query)
    
    def _create_visualization(self, query: str) -> Dict:
        """Create visualization based on query"""
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) >= 2:
            return {
                'interpretation': 'Creating scatter plot of numeric data',
                'data': self.df[numeric_cols[:2]].head(100).to_dict('records'),
                'visualization_type': 'scatter_plot',
                'chart_config': {
                    'x': numeric_cols[0],
                    'y': numeric_cols[1],
                    'title': f'{numeric_cols[1]} vs {numeric_cols[0]}'
                }
            }
        
        return self._general_summary(query)
    
    def _filter_data(self, query: str) -> Dict:
        """Filter data based on query"""
        # Simple filtering - return first 50 rows
        filtered_data = self.df.head(50).to_dict('records')
        
        return {
            'interpretation': 'Filtered dataset (showing first 50 rows)',
            'data': filtered_data,
            'visualization_type': 'table',
            'chart_config': None
        }
    
    def _general_summary(self, query: str) -> Dict:
        """General data summary"""
        summary_data = []
        
        for col in self.df.columns:
            if self.df[col].dtype in ['int64', 'float64']:
                summary_data.append({
                    'column': col,
                    'mean': round(self.df[col].mean(), 2) if not self.df[col].isna().all() else None,
                    'max': self.df[col].max() if not self.df[col].isna().all() else None,
                    'min': self.df[col].min() if not self.df[col].isna().all() else None,
                    'count': self.df[col].count()
                })
            else:
                summary_data.append({
                    'column': col,
                    'unique_values': self.df[col].nunique(),
                    'most_common': self.df[col].mode().iloc[0] if len(self.df[col].mode()) > 0 else None,
                    'count': self.df[col].count()
                })
        
        return {
            'interpretation': 'General data summary statistics',
            'data': summary_data,
            'visualization_type': 'table',
            'chart_config': None
        }
    
    def _get_top_values(self, query: str) -> Dict:
        """Get top N values from dataset"""
        # Extract number if present
        import re
        numbers = re.findall(r'\d+', query)
        n = int(numbers[0]) if numbers else 10
        
        # Find numeric column
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            top_data = self.df.nlargest(n, col).to_dict('records')
            
            return {
                'interpretation': f'Showing top {n} records by {col}',
                'data': top_data,
                'visualization_type': 'bar_chart',
                'chart_config': {
                    'x': self.df.columns[0],
                    'y': col,
                    'title': f'Top {n} by {col}'
                }
            }
        
        return self._general_summary(query)
    
    def _group_analysis(self, query: str) -> Dict:
        """Group data analysis"""
        # Simple grouping by first categorical column
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            group_col = categorical_cols[0]
            value_col = numeric_cols[0]
            
            grouped = self.df.groupby(group_col)[value_col].agg(['sum', 'mean', 'count']).reset_index()
            grouped_data = grouped.to_dict('records')
            
            return {
                'interpretation': f'Data grouped by {group_col} with {value_col} statistics',
                'data': grouped_data,
                'visualization_type': 'bar_chart',
                'chart_config': {
                    'x': group_col,
                    'y': 'sum',
                    'title': f'{value_col} by {group_col}'
                }
            }
        
        return self._general_summary(query)
    
    def _create_visualization(self, query: str) -> Dict:
        """Create visualization based on query"""
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) >= 2:
            return {
                'interpretation': 'Creating scatter plot of numeric data',
                'data': self.df[numeric_cols[:2]].head(100).to_dict('records'),
                'visualization_type': 'scatter_plot',
                'chart_config': {
                    'x': numeric_cols[0],
                    'y': numeric_cols[1],
                    'title': f'{numeric_cols[1]} vs {numeric_cols[0]}'
                }
            }
        
        return self._general_summary(query)
    
    def _filter_data(self, query: str) -> Dict:
        """Filter data based on query"""
        # Simple filtering - return first 50 rows
        filtered_data = self.df.head(50).to_dict('records')
        
        return {
            'interpretation': 'Filtered dataset (showing first 50 rows)',
            'data': filtered_data,
            'visualization_type': 'table',
            'chart_config': None
        }
    
    def _general_summary(self, query: str) -> Dict:
        """General data summary"""
        summary_data = []
        
        for col in self.df.columns:
            if self.df[col].dtype in ['int64', 'float64']:
                summary_data.append({
                    'column': col,
                    'mean': round(self.df[col].mean(), 2) if not self.df[col].isna().all() else None,
                    'max': self.df[col].max() if not self.df[col].isna().all() else None,
                    'min': self.df[col].min() if not self.df[col].isna().all() else None,
                    'count': self.df[col].count()
                })
            else:
                summary_data.append({
                    'column': col,
                    'unique_values': self.df[col].nunique(),
                    'most_common': self.df[col].mode().iloc[0] if len(self.df[col].mode()) > 0 else None,
                    'count': self.df[col].count()
                })
        
        return {
            'interpretation': 'General data summary statistics',
            'data': summary_data,
            'visualization_type': 'table',
            'chart_config': None
        }

# Enhanced query processing with web search capability
class QueryProcessor:
    def __init__(self):
        self.ddgs = DDGS()
    
    async def enhance_query_with_context(self, query: str, data_context: str) -> str:
        """Enhance user query with additional context using web search if needed"""
        
        # Check if query might benefit from external context
        context_keywords = ['trend', 'benchmark', 'industry', 'market', 'compare', 'standard']
        needs_context = any(keyword in query.lower() for keyword in context_keywords)
        
        if needs_context:
            try:
                # Search for relevant context
                search_results = list(self.ddgs.text(query + " data analysis trends", max_results=3))
                
                if search_results:
                    context_info = "\n".join([f"- {r['title']}: {r['body'][:200]}..." for r in search_results])
                    
                    enhanced_prompt = f"""
Original Query: {query}

Dataset Context: {data_context}

External Context (for reference):
{context_info}

Please analyze the data considering both the dataset and any relevant external context.
"""
                    return enhanced_prompt
            except Exception as e:
                print(f"Context search failed: {e}")
        
        return query

# Configuration class for adaptable settings
class AnalysisConfig:
    def __init__(self):
        self.ai_model = os.getenv("AI_MODEL", "openai/gpt-4.1")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "1000"))
        self.temperature = float(os.getenv("AI_TEMPERATURE", "0.3"))
        self.enable_web_search = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"
        self.max_data_rows = int(os.getenv("MAX_DATA_ROWS", "10000"))
        self.supported_file_types = os.getenv("SUPPORTED_FILE_TYPES", "csv,xlsx,json").split(",")

config = AnalysisConfig()

# FastAPI endpoints
@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV file for analysis"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Read CSV content
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Generate file ID
        file_id = f"csv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store in memory (use database in production)
        uploaded_files[file_id] = {
            'filename': file.filename,
            'dataframe': df,
            'upload_time': datetime.now(),
            'row_count': len(df),
            'column_count': len(df.columns)
        }
        
        return {
            'file_id': file_id,
            'filename': file.filename,
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'preview': df.head().to_dict('records')
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")

@app.post("/test-db-connection/")
async def test_db_connection(connection: DatabaseConnection):
    """Test database connection"""
    try:
        engine = get_db_connection(connection.connection_string)
        
        # Test query
        with engine.connect() as conn:
            result = conn.execute("SELECT 1 as test")
            result.fetchone()
        
        return {"status": "success", "message": "Database connection successful"}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Connection failed: {str(e)}")

@app.post("/analyze/", response_model=AnalysisResult)
async def analyze_data(query: NaturalLanguageQuery):
    """Analyze data using AI-powered natural language processing"""
    try:
        df = None
        
        # Get data source
        if query.data_source == 'csv':
            if not query.file_id or query.file_id not in uploaded_files:
                raise HTTPException(status_code=400, detail="File not found")
            df = uploaded_files[query.file_id]['dataframe']
            
        elif query.data_source == 'db':
            if not query.connection_string:
                raise HTTPException(status_code=400, detail="Database connection string required")
            
            engine = get_db_connection(query.connection_string)
            
            with engine.connect() as conn:
                # Get table names
                if 'postgresql' in query.connection_string:
                    tables_query = "SELECT tablename FROM pg_tables WHERE schemaname = 'public';"
                elif 'mysql' in query.connection_string:
                    tables_query = "SHOW TABLES;"
                else:  # SQLite
                    tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
                
                tables = pd.read_sql(tables_query, conn)
                
                if len(tables) > 0:
                    table_name = tables.iloc[0, 0]
                    df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT {config.max_data_rows}", conn)
                else:
                    raise HTTPException(status_code=400, detail="No tables found in database")
        
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="No data available for analysis")
        
        # Initialize AI analyzer and query processor
        analyzer = AIDataAnalyzer(df)
        query_processor = QueryProcessor()
        
        # Enhance query with context if enabled
        if config.enable_web_search:
            enhanced_query = await query_processor.enhance_query_with_context(
                query.query, 
                analyzer.column_info
            )
        else:
            enhanced_query = query.query
        
        # Perform AI-powered analysis
        analysis_result = await analyzer.analyze_query(enhanced_query)
        
        return AnalysisResult(
            query=query.query,
            interpretation=analysis_result['interpretation'],
            data=analysis_result['data'],
            visualization_type=analysis_result['visualization_type'],
            chart_config=analysis_result.get('chart_config'),
            sql_query=analysis_result.get('pandas_code')  # Show the generated pandas code
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/chat/")
async def chat_with_ai(query: str, context: Optional[str] = None):
    """Direct chat interface with AI for general questions"""
    try:
        system_prompt = """You are a helpful data analysis assistant. 
        Answer questions about data analysis, statistics, and provide insights.
        Keep responses concise and practical."""
        
        if context:
            system_prompt += f"\n\nContext: {context}"
        
        response = ai_client.complete(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=query)
            ],
            model=config.ai_model,
            max_tokens=config.max_tokens,
            temperature=config.temperature
        )
        
        return {
            "response": response.choices[0].message.content,
            "model": config.ai_model
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/config/")
async def get_config():
    """Get current configuration settings"""
    return {
        "ai_model": config.ai_model,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "enable_web_search": config.enable_web_search,
        "max_data_rows": config.max_data_rows,
        "supported_file_types": config.supported_file_types
    }

@app.get("/files/")
async def list_uploaded_files():
    """List all uploaded files"""
    files_info = {}
    for file_id, file_data in uploaded_files.items():
        files_info[file_id] = {
            'filename': file_data['filename'],
            'upload_time': file_data['upload_time'],
            'rows': file_data['row_count'],
            'columns': file_data['column_count']
        }
    return files_info

@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete uploaded file"""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    del uploaded_files[file_id]
    return {"message": "File deleted successfully"}

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

# Mount Chainlit app (optional - for direct Chainlit interface)
# app.mount("/chainlit", chainlit_app)

if __name__ == "__main__":
    # Check if API key is configured
    if not token:
        print("WARNING: GITHUB_MODELS_KEY not found in environment variables")
        print("Please set your API key in .env file")
    
    print(f"Starting server with AI model: {config.ai_model}")
    print(f"Web search enabled: {config.enable_web_search}")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)