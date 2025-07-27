
#!/usr/bin/env python3
"""
Complete SQL Export Script for TribexAlpha PostgreSQL Database
Exports all tables, data, and structure as a .sql file
"""

import os
import psycopg
from datetime import datetime

def export_database_to_sql():
    """Export entire PostgreSQL database to .sql file"""
    
    print("🔄 Starting PostgreSQL database export to SQL...")
    
    # Get database connection
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ DATABASE_URL environment variable not found!")
        return False
    
    try:
        conn = psycopg.connect(database_url)
        cursor = conn.cursor()
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        sql_filename = f"tribexalpha_database_backup_{timestamp}.sql"
        
        print(f"📝 Creating SQL dump: {sql_filename}")
        
        with open(sql_filename, 'w') as sql_file:
            # Write header
            sql_file.write("-- TribexAlpha PostgreSQL Database Backup\n")
            sql_file.write(f"-- Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            sql_file.write("-- Database: TribexAlpha Trading Platform\n\n")
            
            sql_file.write("SET client_encoding = 'UTF8';\n")
            sql_file.write("SET standard_conforming_strings = on;\n")
            sql_file.write("SET check_function_bodies = false;\n")
            sql_file.write("SET xmloption = content;\n")
            sql_file.write("SET client_min_messages = warning;\n\n")
            
            # Get all tables
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name
            """)
            tables = cursor.fetchall()
            
            print(f"📊 Found {len(tables)} tables to export")
            
            for (table_name,) in tables:
                print(f"  📋 Exporting table: {table_name}")
                
                # Get table schema
                cursor.execute(f"""
                    SELECT column_name, data_type, is_nullable, column_default, character_maximum_length
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}' 
                    ORDER BY ordinal_position
                """)
                columns = cursor.fetchall()
                
                # Create table structure
                sql_file.write(f"--\n-- Table structure for table `{table_name}`\n--\n\n")
                sql_file.write(f"DROP TABLE IF EXISTS {table_name};\n")
                sql_file.write(f"CREATE TABLE {table_name} (\n")
                
                column_definitions = []
                for col_name, data_type, is_nullable, default_val, char_length in columns:
                    col_def = f"  {col_name} "
                    
                    # Handle data types
                    if data_type == 'character varying':
                        if char_length:
                            col_def += f"VARCHAR({char_length})"
                        else:
                            col_def += "VARCHAR(255)"
                    elif data_type == 'bytea':
                        col_def += "BYTEA"
                    elif data_type == 'integer':
                        col_def += "INTEGER"
                    elif data_type == 'timestamp without time zone':
                        col_def += "TIMESTAMP"
                    elif data_type == 'jsonb':
                        col_def += "JSONB"
                    else:
                        col_def += data_type.upper()
                    
                    # Handle nullable
                    if is_nullable == 'NO':
                        col_def += " NOT NULL"
                    
                    # Handle defaults
                    if default_val:
                        if 'CURRENT_TIMESTAMP' in str(default_val):
                            col_def += " DEFAULT CURRENT_TIMESTAMP"
                        elif 'nextval' in str(default_val):
                            col_def += " PRIMARY KEY"
                    
                    column_definitions.append(col_def)
                
                sql_file.write(",\n".join(column_definitions))
                sql_file.write("\n);\n\n")
                
                # Get primary keys and constraints
                cursor.execute(f"""
                    SELECT constraint_name, constraint_type 
                    FROM information_schema.table_constraints 
                    WHERE table_name = '{table_name}' AND constraint_type = 'PRIMARY KEY'
                """)
                constraints = cursor.fetchall()
                
                for constraint_name, constraint_type in constraints:
                    if constraint_type == 'PRIMARY KEY':
                        cursor.execute(f"""
                            SELECT column_name 
                            FROM information_schema.key_column_usage 
                            WHERE constraint_name = '{constraint_name}'
                        """)
                        pk_columns = cursor.fetchall()
                        if pk_columns and table_name != 'trained_models':  # trained_models has SERIAL
                            pk_cols = ", ".join([col[0] for col in pk_columns])
                            sql_file.write(f"ALTER TABLE {table_name} ADD PRIMARY KEY ({pk_cols});\n\n")
                
                # Export data
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                if row_count > 0:
                    print(f"    📊 Exporting {row_count} rows from {table_name}")
                    sql_file.write(f"--\n-- Data for table `{table_name}`\n--\n\n")
                    
                    # Get all data
                    cursor.execute(f"SELECT * FROM {table_name}")
                    rows = cursor.fetchall()
                    
                    # Get column names for INSERT
                    cursor.execute(f"""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = '{table_name}' 
                        ORDER BY ordinal_position
                    """)
                    column_names = [col[0] for col in cursor.fetchall()]
                    
                    if rows:
                        # Write INSERT statements
                        for row in rows:
                            values = []
                            for i, value in enumerate(row):
                                if value is None:
                                    values.append("NULL")
                                elif isinstance(value, str):
                                    # Escape single quotes
                                    escaped_value = value.replace("'", "''")
                                    values.append(f"'{escaped_value}'")
                                elif isinstance(value, bytes):
                                    # Handle bytea data
                                    hex_value = value.hex()
                                    values.append(f"'\\x{hex_value}'")
                                elif isinstance(value, (int, float)):
                                    values.append(str(value))
                                else:
                                    # Handle other types (timestamps, etc.)
                                    values.append(f"'{str(value)}'")
                            
                            col_list = ", ".join(column_names)
                            val_list = ", ".join(values)
                            sql_file.write(f"INSERT INTO {table_name} ({col_list}) VALUES ({val_list});\n")
                    
                    sql_file.write("\n")
                else:
                    print(f"    📝 Table {table_name} is empty")
                    sql_file.write(f"-- Table {table_name} contains no data\n\n")
            
            # Write footer
            sql_file.write("--\n-- End of TribexAlpha Database Backup\n--\n")
        
        # Get file size
        file_size = os.path.getsize(sql_filename)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"✅ SQL export completed successfully!")
        print(f"📄 File: {sql_filename}")
        print(f"📦 Size: {file_size_mb:.2f} MB")
        print(f"🗃️ Tables exported: {len(tables)}")
        
        # Show summary
        cursor.execute("SELECT COUNT(*) FROM ohlc_datasets")
        dataset_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM trained_models") 
        model_count = cursor.fetchone()[0]
        
        print(f"\n📊 Export Summary:")
        print(f"   - Datasets: {dataset_count}")
        print(f"   - Trained Models: {model_count}")
        print(f"   - Complete SQL backup ready for import")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {str(e)}")
        return False

def main():
    """Main export function"""
    print("🚀 TribexAlpha SQL Database Export")
    print("=" * 50)
    
    success = export_database_to_sql()
    
    if success:
        print("\n✅ Ready for Supabase import!")
        print("📋 Next steps:")
        print("   1. Download the generated .sql file")
        print("   2. Import to Supabase using their SQL editor")
        print("   3. Update your DATABASE_URL to point to Supabase")
    else:
        print("\n❌ Export failed. Check the error messages above.")

if __name__ == "__main__":
    main()
