from datetime import datetime, timedelta

def calculate_date_range():
    today = datetime.now()
    thirty_days_ago = (today - timedelta(days=29)).strftime("%Y-%m-%d")
    return thirty_days_ago, today.strftime("%Y-%m-%d")
