import pandas as pd
import pyodbc
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pathlib import Path
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class QueryResult:
    question: str
    understanding: str
    sql_query: str
    results: pd.DataFrame
    explanation: str
    success: bool

class RagDB:
    def __init__(self):
        self.conn_str = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=localhost;"
            "DATABASE=AdventureWorksDW2022;"
            "Trusted_Connection=yes;"
        )
        
        print("🚀 Initializing RagDB with LLM...")
        
        # Initialize embedding model
        print("📥 Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize local LLM for query understanding and explanation
        print("🧠 Loading local LLM...")
        self._setup_llm()
        
        # Setup vector store
        print("🗄️ Setting up vector database...")
        Path("vectordb").mkdir(exist_ok=True)
        client = chromadb.PersistentClient(path="vectordb")
        self.collection = client.get_or_create_collection("db_knowledge")
        
        # Extract database knowledge
        self.db_schema = {}
        self._extract_database_knowledge()
        self._build_knowledge_base()
        
        print("✅ RagDB ready with LLM integration")
    
    def _setup_llm(self):
        """Setup local LLM for query understanding and response generation"""
        try:
            # Use a smaller, efficient model that works well locally
            model_name = "microsoft/DialoGPT-small"  # Fast and good for conversations
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm_model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create text generation pipeline
            self.text_generator = pipeline(
                "text-generation",
                model=self.llm_model,
                tokenizer=self.tokenizer,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            print("✅ Local LLM loaded successfully")
            
        except Exception as e:
            print(f"⚠️ LLM setup failed, using fallback: {e}")
            self.text_generator = None
    
    def _extract_database_knowledge(self):
        """Extract comprehensive database schema"""
        print("🔍 Extracting database schema...")
        
        conn = pyodbc.connect(self.conn_str)
        cursor = conn.cursor()
        
        # Get all relevant tables
        cursor.execute("""
            SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_SCHEMA = 'dbo'
            AND (TABLE_NAME LIKE 'Dim%' OR TABLE_NAME LIKE 'Fact%')
            ORDER BY TABLE_NAME
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        
        for table_name in tables:
            try:
                # Get columns
                cursor.execute(f"""
                    SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_NAME = '{table_name}'
                    ORDER BY ORDINAL_POSITION
                """)
                columns = cursor.fetchall()
                
                # Get sample data
                cursor.execute(f"SELECT TOP 2 * FROM dbo.{table_name}")
                samples = cursor.fetchall()
                col_names = [desc[0] for desc in cursor.description]
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM dbo.{table_name}")
                row_count = cursor.fetchone()[0]
                
                self.db_schema[table_name] = {
                    'columns': [(col[0], col[1]) for col in columns],
                    'sample_data': [dict(zip(col_names[:5], row[:5])) for row in samples[:1]],
                    'row_count': row_count,
                    'description': self._generate_table_description(table_name, columns, row_count)
                }
                
                print(f"   📊 {table_name}: {len(columns)} columns, {row_count:,} rows")
                
            except Exception as e:
                print(f"   ⚠️ Error with {table_name}: {e}")
        
        conn.close()
    
    def _generate_table_description(self, table_name: str, columns: List, row_count: int) -> str:
        """Generate intelligent table descriptions"""
        
        descriptions = {
            'DimCustomer': 'Customer master data containing customer demographics, contact information, and geographic details',
            'DimProduct': 'Product catalog with product names, categories, subcategories, and pricing information',
            'DimDate': 'Date dimension table for time-based analysis with calendar and fiscal year information',
            'DimGeography': 'Geographic hierarchy with countries, states, regions, and cities',
            'DimEmployee': 'Employee information including names, titles, departments, and organizational hierarchy',
            'DimReseller': 'Business partner and reseller information for channel sales',
            'FactInternetSales': 'Internet sales transactions with sales amounts, quantities, and customer details',
            'FactResellerSales': 'Reseller sales transactions through business partners',
            'FactProductInventory': 'Product inventory levels and movement tracking'
        }
        
        base_desc = descriptions.get(table_name, f"Database table {table_name}")
        key_columns = [col[0] for col in columns if 'key' in col[0].lower()]
        measure_columns = [col[0] for col in columns if any(term in col[0].lower() 
                          for term in ['amount', 'quantity', 'cost', 'price', 'count'])]
        
        description = f"{base_desc}. Contains {row_count:,} records"
        
        if key_columns:
            description += f" with key fields: {', '.join(key_columns[:3])}"
        
        if measure_columns:
            description += f" and measures: {', '.join(measure_columns[:3])}"
        
        return description
    
    def _build_knowledge_base(self):
        """Build vector knowledge base"""
        if self.collection.count() > 0:
            print(f"📚 Knowledge base exists ({self.collection.count()} items)")
            return
        
        print("📚 Building knowledge base...")
        
        documents = []
        metadatas = []
        ids = []
        
        # Add table knowledge
        for table_name, info in self.db_schema.items():
            # Table description
            documents.append(info['description'])
            metadatas.append({'type': 'table', 'table': table_name})
            ids.append(f"table_{table_name}")
            
            # Column information
            col_info = f"Table {table_name} has columns: "
            col_info += ", ".join([f"{col[0]} ({col[1]})" for col in info['columns'][:10]])
            documents.append(col_info)
            metadatas.append({'type': 'columns', 'table': table_name})
            ids.append(f"cols_{table_name}")
            
            # Sample data context
            if info['sample_data']:
                sample_info = f"Sample data from {table_name}: {str(info['sample_data'][0])}"
                documents.append(sample_info)
                metadatas.append({'type': 'sample', 'table': table_name})
                ids.append(f"sample_{table_name}")
        
        # Add query patterns
        patterns = [
            "For customer analysis use DimCustomer joined with FactInternetSales on CustomerKey",
            "For product performance use DimProduct joined with FactInternetSales on ProductKey",
            "For time-based analysis use DimDate joined with fact tables on DateKey",
            "For geographic analysis use DimGeography joined with DimCustomer on GeographyKey",
            "For employee information query DimEmployee table directly",
            "To find top customers: GROUP BY customer and ORDER BY SUM(SalesAmount) DESC",
            "To find best products: GROUP BY product and ORDER BY SUM(SalesAmount) DESC",
            "For yearly trends: GROUP BY CalendarYear from DimDate"
        ]
        
        for i, pattern in enumerate(patterns):
            documents.append(pattern)
            metadatas.append({'type': 'pattern', 'pattern_id': i})
            ids.append(f"pattern_{i}")
        
        # Generate embeddings
        print(f"🧠 Generating embeddings for {len(documents)} items...")
        embeddings = self.embedder.encode(documents)
        
        # Store in vector database
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Knowledge base built with {len(documents)} items")
    
    def _understand_question(self, question: str) -> str:
        """Use LLM to better understand the user's question"""
        
        if not self.text_generator:
            return f"User is asking about: {question}"
        
        prompt = f"""
        Analyze this database question and explain what the user wants to know:
        Question: "{question}"
        
        The user wants to understand:
        """
        
        try:
            response = self.text_generator(
                prompt,
                max_length=len(prompt.split()) + 30,
                num_return_sequences=1,
                temperature=0.3
            )
            
            generated_text = response[0]['generated_text']
            understanding = generated_text.replace(prompt, "").strip()
            
            return understanding if understanding else f"Analysis of: {question}"
            
        except Exception as e:
            return f"User is asking about: {question}"
    
    def _get_relevant_context(self, question: str, top_k: int = 5) -> List[str]:
        """Retrieve relevant context from knowledge base"""
        
        # Generate query embedding
        query_embedding = self.embedder.encode([question])
        
        # Search vector database
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        return results['documents'][0] if results['documents'] else []
    
    def _generate_smart_sql(self, question: str, context: List[str]) -> str:
        """Generate SQL using question understanding and context"""
        
        question_lower = question.lower()
        
        # Enhanced pattern matching with context awareness
        if any(phrase in question_lower for phrase in ['top customer', 'best customer', 'customer.*sales']):
            return """
            SELECT TOP 10 
                c.CustomerKey,
                c.FirstName + ' ' + c.LastName AS CustomerName,
                c.City,
                g.EnglishCountryRegionName AS Country,
                SUM(f.SalesAmount) AS TotalSales,
                COUNT(f.SalesOrderNumber) AS OrderCount
            FROM DimCustomer c
            INNER JOIN FactInternetSales f ON c.CustomerKey = f.CustomerKey
            INNER JOIN DimGeography g ON c.GeographyKey = g.GeographyKey
            GROUP BY c.CustomerKey, c.FirstName, c.LastName, c.City, g.EnglishCountryRegionName
            ORDER BY TotalSales DESC
            """
        
        elif any(phrase in question_lower for phrase in ['best product', 'top product', 'product.*sales']):
            return """
            SELECT TOP 10
                p.EnglishProductName AS ProductName,
                pc.EnglishProductCategoryName AS Category,
                SUM(f.SalesAmount) AS TotalSales,
                SUM(f.OrderQuantity) AS TotalQuantity,
                COUNT(DISTINCT f.CustomerKey) AS UniqueCustomers
            FROM DimProduct p
            INNER JOIN FactInternetSales f ON p.ProductKey = f.ProductKey
            INNER JOIN DimProductSubcategory psc ON p.ProductSubcategoryKey = psc.ProductSubcategoryKey
            INNER JOIN DimProductCategory pc ON psc.ProductCategoryKey = pc.ProductCategoryKey
            GROUP BY p.ProductKey, p.EnglishProductName, pc.EnglishProductCategoryName
            ORDER BY TotalSales DESC
            """
        
        elif any(phrase in question_lower for phrase in ['employee', 'staff', 'worker', 'best employee']):
            return """
            SELECT TOP 10
                e.FirstName + ' ' + e.LastName AS EmployeeName,
                e.Title,
                e.DepartmentName,
                e.BaseRate,
                e.VacationHours,
                e.SickLeaveHours,
                e.HireDate
            FROM DimEmployee e
            WHERE e.Status = 'Current'
            ORDER BY e.BaseRate DESC
            """
        
        elif any(phrase in question_lower for phrase in ['sales.*year', 'yearly', 'annual', 'trend']):
            return """
            SELECT 
                d.CalendarYear AS Year,
                SUM(f.SalesAmount) AS YearlySales,
                COUNT(DISTINCT f.SalesOrderNumber) AS OrderCount,
                COUNT(DISTINCT f.CustomerKey) AS UniqueCustomers,
                AVG(f.SalesAmount) AS AvgOrderValue
            FROM DimDate d
            INNER JOIN FactInternetSales f ON d.DateKey = f.OrderDateKey
            GROUP BY d.CalendarYear
            ORDER BY d.CalendarYear
            """
        
        elif any(phrase in question_lower for phrase in ['country', 'geographic', 'location', 'region']):
            return """
            SELECT 
                g.EnglishCountryRegionName AS Country,
                COUNT(DISTINCT c.CustomerKey) AS CustomerCount,
                SUM(f.SalesAmount) AS TotalSales,
                AVG(f.SalesAmount) AS AvgOrderValue
            FROM DimGeography g
            INNER JOIN DimCustomer c ON g.GeographyKey = c.GeographyKey
            INNER JOIN FactInternetSales f ON c.CustomerKey = f.CustomerKey
            GROUP BY g.EnglishCountryRegionName
            ORDER BY TotalSales DESC
            """
        
        elif any(phrase in question_lower for phrase in ['total sales', 'sum.*sales', 'revenue']):
            return """
            SELECT 
                SUM(SalesAmount) AS TotalSales,
                COUNT(DISTINCT SalesOrderNumber) AS TotalOrders,
                COUNT(DISTINCT CustomerKey) AS UniqueCustomers,
                AVG(SalesAmount) AS AvgOrderValue
            FROM FactInternetSales
            """
        
        elif any(phrase in question_lower for phrase in ['customer count', 'how many customer']):
            return """
            SELECT 
                COUNT(*) AS TotalCustomers,
                COUNT(CASE WHEN Gender = 'M' THEN 1 END) AS MaleCustomers,
                COUNT(CASE WHEN Gender = 'F' THEN 1 END) AS FemaleCustomers,
                COUNT(DISTINCT GeographyKey) AS UniqueLocations
            FROM DimCustomer
            """
        
        else:
            # Default to recent sales with more detail
            return """
            SELECT TOP 10
                c.FirstName + ' ' + c.LastName AS Customer,
                p.EnglishProductName AS Product,
                f.SalesAmount,
                f.OrderQuantity,
                d.FullDateAlternateKey AS OrderDate
            FROM FactInternetSales f
            INNER JOIN DimCustomer c ON f.CustomerKey = c.CustomerKey
            INNER JOIN DimProduct p ON f.ProductKey = p.ProductKey
            INNER JOIN DimDate d ON f.OrderDateKey = d.DateKey
            ORDER BY f.SalesOrderNumber DESC
            """
    
    def _execute_query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query safely"""
        try:
            conn = pyodbc.connect(self.conn_str)
            df = pd.read_sql(sql, conn)
            conn.close()
            return df
        except Exception as e:
            print(f"❌ SQL Error: {e}")
            return pd.DataFrame()
    
    def _explain_results(self, question: str, results: pd.DataFrame, sql: str) -> str:
        """Generate natural language explanation of results"""
        
        if results.empty:
            return "No data was found for your question. This could mean the criteria didn't match any records in the database."
        
        # Generate contextual explanation based on question type
        question_lower = question.lower()
        
        if 'customer' in question_lower and 'top' in question_lower:
            top_customer = results.iloc[0]
            explanation = f"I found the top customers by sales volume. The highest spending customer is "
            explanation += f"{top_customer.get('CustomerName', 'Unknown')} with "
            explanation += f"${top_customer.get('TotalSales', 0):,.2f} in total sales. "
            explanation += f"The results show {len(results)} customers with their sales totals, order counts, and locations."
            
        elif 'product' in question_lower and 'best' in question_lower:
            top_product = results.iloc[0]
            explanation = f"Here are the best-selling products by revenue. The top performer is "
            explanation += f"'{top_product.get('ProductName', 'Unknown')}' which generated "
            explanation += f"${top_product.get('TotalSales', 0):,.2f} in sales. "
            explanation += f"I've included the product category and customer reach for each item."
            
        elif 'employee' in question_lower:
            if not results.empty:
                explanation = f"I found {len(results)} employees in the database. "
                explanation += f"The results are sorted by base salary rate, showing employee details "
                explanation += f"including their department, title, and employment status."
            else:
                explanation = "No employee data was found matching your criteria."
                
        elif 'year' in question_lower or 'trend' in question_lower:
            if len(results) > 1:
                first_year = results.iloc[0]
                last_year = results.iloc[-1]
                explanation = f"Here's the sales trend over {len(results)} years. "
                explanation += f"Sales started at ${first_year.get('YearlySales', 0):,.0f} in {first_year.get('Year', 'Unknown')} "
                explanation += f"and reached ${last_year.get('YearlySales', 0):,.0f} in {last_year.get('Year', 'Unknown')}. "
                explanation += f"The data includes order counts and unique customer metrics for each year."
            else:
                explanation = f"Found sales data for {len(results)} time period(s)."
                
        elif 'country' in question_lower or 'geographic' in question_lower:
            if not results.empty:
                top_country = results.iloc[0]
                explanation = f"Here's the geographic sales breakdown. {top_country.get('Country', 'Unknown')} "
                explanation += f"leads with ${top_country.get('TotalSales', 0):,.2f} in sales "
                explanation += f"from {top_country.get('CustomerCount', 0)} customers. "
                explanation += f"The results show performance across {len(results)} countries/regions."
            else:
                explanation = "No geographic sales data was found."
                
        elif 'total' in question_lower and 'sales' in question_lower:
            if not results.empty:
                total = results.iloc[0]
                explanation = f"The total sales amount is ${total.get('TotalSales', 0):,.2f} "
                explanation += f"across {total.get('TotalOrders', 0)} orders "
                explanation += f"from {total.get('UniqueCustomers', 0)} unique customers. "
                explanation += f"The average order value is ${total.get('AvgOrderValue', 0):,.2f}."
            else:
                explanation = "No sales data was found."
                
        else:
            explanation = f"I found {len(results)} records matching your question. "
            explanation += f"The results show {', '.join(results.columns[:3])} and other relevant details. "
            explanation += f"You can see the specific data values in the table above."
        
        return explanation
    
    def ask(self, question: str) -> QueryResult:
        """Main interface - ask any question about the database"""
        
        print(f"\n🤔 Question: {question}")
        
        # Step 1: Understand the question using LLM
        print("🧠 Understanding question...")
        understanding = self._understand_question(question)
        print(f"💭 Understanding: {understanding}")
        
        # Step 2: Get relevant context
        print("📚 Retrieving relevant context...")
        context = self._get_relevant_context(question)
        print(f"📖 Found {len(context)} relevant knowledge items")
        
        # Step 3: Generate intelligent SQL
        print("🔍 Generating SQL query...")
        sql = self._generate_smart_sql(question, context)
        print("Generated SQL:")
        print("─" * 50)
        print(sql.strip())
        print("─" * 50)
        
        # Step 4: Execute query
        print("⚙️ Executing query...")
        results = self._execute_query(sql)
        
        # Step 5: Generate explanation
        explanation = self._explain_results(question, results, sql)
        
        if not results.empty:
            print(f"✅ Results ({len(results)} rows):")
            print(results.to_string(index=False, max_rows=10))
            print(f"\n💡 Explanation: {explanation}")
            success = True
        else:
            print("❌ No results found")
            print(f"💡 Explanation: {explanation}")
            success = False
        
        return QueryResult(
            question=question,
            understanding=understanding,
            sql_query=sql.strip(),
            results=results,
            explanation=explanation,
            success=success
        )
    
    def interactive(self):
        """Interactive mode with enhanced conversations"""
        print("\n💬 RagDB Interactive Mode")
        print("Ask me anything about your AdventureWorks database!")
        print("I'll use AI to understand your questions and explain the results.")
        print("Type 'quit' to exit, 'help' for examples")
        print("─" * 60)
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Thanks for using RagDB!")
                    break
                
                elif question.lower() == 'help':
                    self._show_examples()
                
                elif question:
                    self.ask(question)
                
            except KeyboardInterrupt:
                print("\n👋 Thanks for using RagDB!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _show_examples(self):
        """Show example queries with explanations"""
        examples = [
            ("Show me the top 10 customers by sales", "Find highest-value customers"),
            ("What are the best selling products?", "Analyze product performance"),
            ("Who is the best employee?", "Review employee information"),
            ("Show me sales trends by year", "Analyze sales over time"),
            ("Which countries have the highest sales?", "Geographic sales analysis"),
            ("What is our total sales revenue?", "Overall business metrics"),
            ("How many customers do we have?", "Customer base size")
        ]
        
        print("\n💡 Example questions:")
        for i, (question, description) in enumerate(examples, 1):
            print(f"   {i}. {question}")
            print(f"      → {description}")

# Run the system
if __name__ == "__main__":
    # Initialize RagDB with LLM
    ragdb = RagDB()
    
    # Demo queries
    demo_questions = [
        "Show me the top 10 customers by sales",
        "Who is the best employee?",
        "What are the best selling products?"
    ]
    
    print("\n🎯 Demo Queries:")
    for question in demo_questions:
        ragdb.ask(question)
    
    # Start interactive mode
    ragdb.interactive()