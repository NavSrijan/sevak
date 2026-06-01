#!/usr/bin/env python3
import os
import sys
import argparse
from dotenv import load_dotenv
import psycopg

def get_db_url():
    load_dotenv()
    # Prefer PG_DIRECT_URL since it bypasses connection poolers and is standard for administrative tasks
    url = os.getenv("PG_DIRECT_URL")
    if not url:
        # Fallback to PG_DATABASE_URL if PG_DIRECT_URL is not set
        url = os.getenv("PG_DATABASE_URL")
    return url

def truncate_database(force=False):
    db_url = get_db_url()
    if not db_url:
        print("Error: Neither PG_DIRECT_URL nor PG_DATABASE_URL is set in environment.")
        sys.exit(1)

    print("Connecting to database...")
    try:
        # Connect to the database
        conn = psycopg.connect(db_url)
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        sys.exit(1)

    try:
        with conn.cursor() as cur:
            # Query all user tables in the public schema
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                  AND table_type = 'BASE TABLE'
                  AND table_name != 'alembic_version';
            """)
            tables = [row[0] for row in cur.fetchall()]

            if not tables:
                print("No tables found to truncate in the 'public' schema.")
                return

            print("\nThe following tables will be truncated:")
            for table in tables:
                print(f"  - {table}")
            print()

            if not force:
                confirm = input("Are you sure you want to truncate these tables? This will delete ALL data! (y/N): ")
                if confirm.lower() not in ('y', 'yes'):
                    print("Truncation cancelled.")
                    return

            print("Truncating tables...")
            # Generate truncate statement
            table_list = ", ".join(f'"{table}"' for table in tables)
            cur.execute(f"TRUNCATE TABLE {table_list} RESTART IDENTITY CASCADE;")
            conn.commit()
            print("Successfully truncated all tables and restarted identities.")

    except Exception as e:
        conn.rollback()
        print(f"Error during truncation: {e}")
        sys.exit(1)
    finally:
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Truncate all user tables in the database.")
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Bypass confirmation prompt and truncate immediately."
    )
    args = parser.parse_args()
    truncate_database(force=args.yes)
