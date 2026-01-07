import json
from pathlib import Path

dataset = [
    {
        "task": "Flatten all nested objects consistently into top-level columns.",
        "input": {
            "id": "INC001",
            "assigned_to": {"id": "u1", "name": "Alice"},
            "opened_by": {"id": "u2", "name": "Bob"}
        },
        "output": {
            "id": "INC001",
            "assigned_to_id": "u1",
            "assigned_to_name": "Alice",
            "opened_by_id": "u2",
            "opened_by_name": "Bob"
        },
        "ground_truth_code": """def transform(data):
    result = {}
    
    def flatten(obj, prefix=''):
        for key, value in obj.items():
            new_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                flatten(value, new_key)
            else:
                result[new_key] = value
    
    flatten(data)
    return result"""
    },
    {
        "task": "Handle inconsistent object flattening: flatten all objects including deeper nested ones.",
        "input": {
            "id": "INC002",
            "assigned_to": {"id": "u1", "name": "Alice"},
            "opened_by_link": {"id": "u2", "name": "Bob"},
            "sys_domain_id": {"value": "global", "display_value": "Global"}
        },
        "output": {
            "id": "INC002",
            "assigned_to_id": "u1",
            "assigned_to_name": "Alice",
            "opened_by_link_id": "u2",
            "opened_by_link_name": "Bob",
            "sys_domain_id_value": "global",
            "sys_domain_id_display_value": "Global"
        },
        "ground_truth_code": """def transform(data):
    result = {}
    
    def flatten(obj, prefix=''):
        for key, value in obj.items():
            new_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                flatten(value, new_key)
            else:
                result[new_key] = value
    
    flatten(data)
    return result"""
    },
    {
        "task": "Recursively flatten nested objects with more than one level of depth.",
        "input": {
            "user": {
                "id": "u1",
                "profile": {"email": "a@test.com", "age": 30}
            }
        },
        "output": {
            "user_id": "u1",
            "user_profile_email": "a@test.com",
            "user_profile_age": 30
        },
        "ground_truth_code": """def transform(data):
    result = {}
    
    def flatten(obj, prefix=''):
        for key, value in obj.items():
            new_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                flatten(value, new_key)
            else:
                result[new_key] = value
    
    flatten(data)
    return result"""
    },
    {
        "task": "Convert array of objects into multiple rows (explode), preserving top-level data.",
        "input": {
            "order_id": "O1",
            "items": [
                {"sku": "S1", "qty": 2},
                {"sku": "S2", "qty": 1}
            ]
        },
        "output": [
            {"order_id": "O1", "items_sku": "S1", "items_qty": 2},
            {"order_id": "O1", "items_sku": "S2", "items_qty": 1}
        ],
        "ground_truth_code": """def transform(data):
    array_field = None
    array_data = None
    base_data = {}
    
    for key, value in data.items():
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            array_field = key
            array_data = value
        else:
            base_data[key] = value
    
    if array_field is None:
        return [data]
    
    result = []
    for item in array_data:
        row = base_data.copy()
        for k, v in item.items():
            row[f"{array_field}_{k}"] = v
        result.append(row)
    
    return result"""
    },
    {
        "task": "Normalize enumeration values to standard lowercase snake_case.",
        "input": {"id": "INC003", "priority": "High Priority"},
        "output": {"id": "INC003", "priority": "high_priority"},
        "ground_truth_code": """def transform(data):
    result = data.copy()
    for key, value in result.items():
        if isinstance(value, str) and not key.endswith('_id') and key not in ['id', 'email']:
            result[key] = value.lower().replace(' ', '_').replace('-', '_')
    return result"""
    },
    {
        "task": "Standardize timestamp formats to UTC ISO-8601.",
        "input": {"id": "INC004", "opened_at": "2024-01-01T10:00:00-05:00"},
        "output": {"id": "INC004", "opened_at": "2024-01-01T15:00:00Z"},
        "ground_truth_code": """def transform(data):
    from datetime import datetime
    result = data.copy()
    for key, value in result.items():
        if isinstance(value, str) and ('T' in value or 'at' in key or 'date' in key):
            try:
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                result[key] = dt.strftime('%Y-%m-%dT%H:%M:%SZ') if dt.utcoffset() else value
                if dt.utcoffset():
                    utc_dt = dt.utctimetuple()
                    from datetime import timezone
                    result[key] = datetime(*utc_dt[:6], tzinfo=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            except:
                pass
    return result"""
    },
    {
        "task": "Coerce numeric-like strings into integer types.",
        "input": {"id": "INC005", "retry_count": "3"},
        "output": {"id": "INC005", "retry_count": 3},
        "ground_truth_code": """def transform(data):
    result = {}
    for key, value in data.items():
        if isinstance(value, str) and value.isdigit():
            result[key] = int(value)
        else:
            result[key] = value
    return result"""
    },
    {
        "task": "Normalize boolean-like values (e.g., 'yes', 'no') to strict true/false.",
        "input": {"id": "INC006", "active": "yes"},
        "output": {"id": "INC006", "active": True},
        "ground_truth_code": """def transform(data):
    bool_map = {
        'yes': True, 'no': False,
        'true': True, 'false': False,
        '1': True, '0': False,
        'y': True, 'n': False
    }
    result = {}
    for key, value in data.items():
        if isinstance(value, str) and value.lower() in bool_map:
            result[key] = bool_map[value.lower()]
        else:
            result[key] = value
    return result"""
    },
    {
        "task": "Create a derived field from existing columns (e.g., extract domain from email).",
        "input": {"id": "INC007", "email": "user@example.com"},
        "output": {
            "id": "INC007",
            "email": "user@example.com",
            "email_domain": "example.com"
        },
        "ground_truth_code": """def transform(data):
    result = data.copy()
    if 'email' in result and isinstance(result['email'], str) and '@' in result['email']:
        result['email_domain'] = result['email'].split('@')[1]
    return result"""
    },
    {
        "task": "Handle missing optional fields without dropping rows.",
        "input": {"id": "INC008", "description": None},
        "output": {"id": "INC008", "description": None},
        "ground_truth_code": """def transform(data):
    return data.copy()"""
    },
    {
        "task": "Apply lookup-based enrichment mapping codes to human-readable labels.",
        "input": {"id": "INC009", "status_code": 2},
        "output": {
            "id": "INC009",
            "status_code": 2,
            "status_label": "in_progress"
        },
        "ground_truth_code": """def transform(data):
    status_lookup = {
        1: "new",
        2: "in_progress",
        3: "resolved",
        4: "closed"
    }
    result = data.copy()
    if 'status_code' in result:
        result['status_label'] = status_lookup.get(result['status_code'], 'unknown')
    return result"""
    },
    {
        "task": "Ensure final output contains no nested objects or arrays.",
        "input": {"id": "INC010", "metadata": {"source": "api", "version": "v1"}},
        "output": {
            "id": "INC010",
            "metadata_source": "api",
            "metadata_version": "v1"
        },
        "ground_truth_code": """def transform(data):
    result = {}
    
    def flatten(obj, prefix=''):
        for key, value in obj.items():
            new_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                flatten(value, new_key)
            else:
                result[new_key] = value
    
    flatten(data)
    return result"""
    },
    {
        "task": "Handle polymorphic field types with consistent type normalization (string to array).",
        "input": {"id": "INC011", "hobbies": "reading"},
        "output": {"id": "INC011", "hobbies": ["reading"]},
        "ground_truth_code": """def transform(data):
    result = data.copy()
    if 'hobbies' in result and isinstance(result['hobbies'], str):
        result['hobbies'] = [result['hobbies']]
    return result"""
    },
    {
        "task": "Convert numeric formats with comma thousand separators.",
        "input": {"id": "INC012", "revenue": "1,234.56"},
        "output": {"id": "INC012", "revenue": 1234.56},
        "ground_truth_code": """def transform(data):
    result = {}
    for key, value in data.items():
        if isinstance(value, str) and ',' in value:
            try:
                result[key] = float(value.replace(',', ''))
            except:
                result[key] = value
        else:
            result[key] = value
    return result"""
    },
    {
        "task": "Handle corrupted numeric fields gracefully with data quality flags.",
        "input": {"id": "INC013", "count": "N/A"},
        "output": {"id": "INC013", "count": None, "data_quality_flag": "invalid_count"},
        "ground_truth_code": """def transform(data):
    result = data.copy()
    if 'count' in result:
        if isinstance(result['count'], str) and not result['count'].replace('.', '').replace('-', '').isdigit():
            result['count'] = None
            result['data_quality_flag'] = 'invalid_count'
    return result"""
    },
    {
        "task": "Handle empty arrays by converting to None.",
        "input": {"id": "INC014", "tags": []},
        "output": {"id": "INC014", "tags": None},
        "ground_truth_code": """def transform(data):
    result = {}
    for key, value in data.items():
        if isinstance(value, list) and len(value) == 0:
            result[key] = None
        else:
            result[key] = value
    return result"""
    },
    {
        "task": "Deduplicate records based on composite keys, keeping latest.",
        "input": [
            {"user_id": "u1", "product_id": "p1", "timestamp": "2024-01-01T10:00:00Z"},
            {"user_id": "u1", "product_id": "p1", "timestamp": "2024-01-01T11:00:00Z"}
        ],
        "output": [
            {"user_id": "u1", "product_id": "p1", "timestamp": "2024-01-01T11:00:00Z"}
        ],
        "ground_truth_code": """def transform(data):
    if not isinstance(data, list):
        return data
    
    seen = {}
    for record in data:
        key = (record.get('user_id'), record.get('product_id'))
        if key not in seen or record.get('timestamp', '') > seen[key].get('timestamp', ''):
            seen[key] = record
    
    return list(seen.values())"""
    },
    {
        "task": "Pivot array of key-value pairs into individual columns.",
        "input": {
            "id": "INC015",
            "attributes": [
                {"key": "color", "value": "red"},
                {"key": "size", "value": "large"}
            ]
        },
        "output": {
            "id": "INC015",
            "attr_color": "red",
            "attr_size": "large"
        },
        "ground_truth_code": """def transform(data):
    result = {}
    for key, value in data.items():
        if key == 'attributes' and isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and 'key' in item and 'value' in item:
                    result[f"attr_{item['key']}"] = item['value']
        else:
            result[key] = value
    return result"""
    },
    {
        "task": "Normalize phone numbers to E.164 format (US numbers).",
        "input": {"id": "INC017", "phone": "(555) 123-4567"},
        "output": {"id": "INC017", "phone": "+15551234567"},
        "ground_truth_code": """def transform(data):
    import re
    result = data.copy()
    if 'phone' in result:
        digits = re.sub(r'\D', '', result['phone'])
        if len(digits) == 10:
            result['phone'] = f"+1{digits}"
    return result"""
    },
    {
        "task": "Split delimited string fields into arrays.",
        "input": {"id": "INC022", "categories": "electronics,gadgets,smartphones"},
        "output": {"id": "INC022", "categories": ["electronics", "gadgets", "smartphones"]},
        "ground_truth_code": """def transform(data):
    result = data.copy()
    if 'categories' in result and isinstance(result['categories'], str):
        result['categories'] = [cat.strip() for cat in result['categories'].split(',')]
    return result"""
    },
    {
        "task": "Aggregate array of metrics into summary statistics.",
        "input": {
            "id": "INC020",
            "daily_sales": [
                {"date": "2024-01-01", "amount": 100},
                {"date": "2024-01-02", "amount": 150},
                {"date": "2024-01-03", "amount": 200}
            ]
        },
        "output": {
            "id": "INC020",
            "total_sales": 450,
            "avg_daily_sales": 150,
            "max_daily_sales": 200,
            "min_daily_sales": 100,
            "sales_days": 3
        },
        "ground_truth_code": """def transform(data):
    result = {'id': data['id']}
    if 'daily_sales' in data:
        amounts = [item['amount'] for item in data['daily_sales']]
        result['total_sales'] = sum(amounts)
        result['avg_daily_sales'] = sum(amounts) // len(amounts)
        result['max_daily_sales'] = max(amounts)
        result['min_daily_sales'] = min(amounts)
        result['sales_days'] = len(amounts)
    return result"""
    },
    {
        "task": "Parse JSON string fields into structured data.",
        "input": {"id": "INC026", "config": "{\"debug\": true, \"timeout\": 30}"},
        "output": {
            "id": "INC026",
            "config_debug": True,
            "config_timeout": 30
        },
        "ground_truth_code": """def transform(data):
    import json
    result = {}
    for key, value in data.items():
        if key == 'config' and isinstance(value, str):
            try:
                config_data = json.loads(value)
                for k, v in config_data.items():
                    result[f"{key}_{k}"] = v
            except:
                result[key] = value
        else:
            result[key] = value
    return result"""
    },
    {
        "task": "Normalize case-insensitive enum values.",
        "input": {"id": "INC027", "status": "In-Progress"},
        "output": {"id": "INC027", "status": "in_progress"},
        "ground_truth_code": """def transform(data):
    result = data.copy()
    if 'status' in result and isinstance(result['status'], str):
        result['status'] = result['status'].lower().replace('-', '_').replace(' ', '_')
    return result"""
    },
    {
        "task": "Handle null propagation in derived calculations.",
        "input": {"id": "INC028", "quantity": None, "price": 10.00},
        "output": {"id": "INC028", "quantity": None, "price": 10.00, "total": None},
        "ground_truth_code": """def transform(data):
    result = data.copy()
    if 'quantity' in result and 'price' in result:
        if result['quantity'] is None or result['price'] is None:
            result['total'] = None
        else:
            result['total'] = result['quantity'] * result['price']
    return result"""
    },
    {
        "task": "Standardize inconsistent date formats to ISO format.",
        "input": {
            "id": "INC033",
            "date1": "01/15/2024",
            "date2": "2024-01-15",
            "date3": "15-Jan-2024"
        },
        "output": {
            "id": "INC033",
            "date1": "2024-01-15",
            "date2": "2024-01-15",
            "date3": "2024-01-15"
        },
        "ground_truth_code": """def transform(data):
    from datetime import datetime
    result = {}
    for key, value in data.items():
        if isinstance(value, str) and any(c in value for c in ['/', '-']) and key.startswith('date'):
            try:
                for fmt in ['%m/%d/%Y', '%Y-%m-%d', '%d-%b-%Y']:
                    try:
                        dt = datetime.strptime(value, fmt)
                        result[key] = dt.strftime('%Y-%m-%d')
                        break
                    except:
                        continue
                else:
                    result[key] = value
            except:
                result[key] = value
        else:
            result[key] = value
    return result"""
    },
    {
        "task": "Apply conditional transformations based on field values (pricing rules).",
        "input": {"id": "INC034", "type": "premium", "base_price": 100},
        "output": {
            "id": "INC034",
            "type": "premium",
            "base_price": 100,
            "final_price": 90,
            "discount_applied": 10
        },
        "ground_truth_code": """def transform(data):
    result = data.copy()
    if 'type' in result and 'base_price' in result:
        if result['type'] == 'premium':
            discount = result['base_price'] * 0.1
            result['discount_applied'] = int(discount)
            result['final_price'] = int(result['base_price'] - discount)
    return result"""
    },
    {
        "task": "Calculate running totals within groups (window function).",
        "input": [
            {"user_id": "u1", "date": "2024-01-01", "amount": 10},
            {"user_id": "u1", "date": "2024-01-02", "amount": 20},
            {"user_id": "u2", "date": "2024-01-01", "amount": 15}
        ],
        "output": [
            {"user_id": "u1", "date": "2024-01-01", "amount": 10, "running_total": 10},
            {"user_id": "u1", "date": "2024-01-02", "amount": 20, "running_total": 30},
            {"user_id": "u2", "date": "2024-01-01", "amount": 15, "running_total": 15}
        ],
        "ground_truth_code": """def transform(data):
    if not isinstance(data, list):
        return data
    
    sorted_data = sorted(data, key=lambda x: (x['user_id'], x['date']))
    result = []
    running_totals = {}
    
    for record in sorted_data:
        user_id = record['user_id']
        if user_id not in running_totals:
            running_totals[user_id] = 0
        running_totals[user_id] += record['amount']
        
        new_record = record.copy()
        new_record['running_total'] = running_totals[user_id]
        result.append(new_record)
    
    return result"""
    },
    {
        "task": "Remove special characters from text, keeping only alphanumeric and spaces.",
        "input": {"id": "INC031", "name": "Café René—™"},
        "output": {"id": "INC031", "name": "Café René"},
        "ground_truth_code": """def transform(data):
    import re
    result = data.copy()
    if 'name' in result and isinstance(result['name'], str):
        result['name'] = re.sub(r'[^\w\s]', '', result['name'], flags=re.UNICODE).strip()
    return result"""
    },
    {
        "task": "Extract fiscal quarter from date.",
        "input": {"id": "INC025", "transaction_date": "2024-03-15"},
        "output": {
            "id": "INC025",
            "transaction_date": "2024-03-15",
            "fiscal_year": 2024,
            "fiscal_quarter": "Q1"
        },
        "ground_truth_code": """def transform(data):
    result = data.copy()
    if 'transaction_date' in result:
        year, month, day = result['transaction_date'].split('-')
        result['fiscal_year'] = int(year)
        month_num = int(month)
        result['fiscal_quarter'] = f"Q{(month_num - 1) // 3 + 1}"
    return result"""
    },
    {
        "task": "Merge duplicate records with last-write-wins conflict resolution.",
        "input": [
            {"id": "u1", "name": "Alice", "email": "alice@old.com", "updated": "2024-01-01T10:00:00Z"},
            {"id": "u1", "name": "Alice Smith", "email": "alice@new.com", "updated": "2024-01-02T10:00:00Z"}
        ],
        "output": {
            "id": "u1",
            "name": "Alice Smith",
            "email": "alice@new.com",
            "updated": "2024-01-02T10:00:00Z"
        },
        "ground_truth_code": """def transform(data):
    if not isinstance(data, list):
        return data
    
    records_by_id = {}
    for record in data:
        rec_id = record['id']
        if rec_id not in records_by_id or record['updated'] > records_by_id[rec_id]['updated']:
            records_by_id[rec_id] = record
    
    return records_by_id[list(records_by_id.keys())[0]]"""
    },
    {
        "task": "Detect and flag outliers in numeric data using simple threshold.",
        "input": [
            {"id": "t1", "response_time_ms": 150},
            {"id": "t2", "response_time_ms": 180},
            {"id": "t3", "response_time_ms": 5000}
        ],
        "output": [
            {"id": "t1", "response_time_ms": 150, "is_outlier": False},
            {"id": "t2", "response_time_ms": 180, "is_outlier": False},
            {"id": "t3", "response_time_ms": 5000, "is_outlier": True}
        ],
        "ground_truth_code": """def transform(data):
    if not isinstance(data, list):
        return data
    
    values = [r['response_time_ms'] for r in data]
    avg = sum(values) / len(values)
    threshold = avg * 3
    
    result = []
    for record in data:
        new_record = record.copy()
        new_record['is_outlier'] = record['response_time_ms'] > threshold
        result.append(new_record)
    
    return result"""
    }
]

output_file = Path("etl_eval_dataset_with_code.json")
with open(output_file, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Dataset with {len(dataset)} test cases written to {output_file}")
print(f"\nEach test case includes:")
print(f"  - task: description of transformation")
print(f"  - input: sample input data")
print(f"  - output: expected output")
print(f"  - ground_truth_code: reference implementation")
print(f"\nTo evaluate, compare predicted_code output vs ground_truth_code output on the input.")