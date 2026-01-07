import json
from pathlib import Path

dataset = [
    # Embedding Generation Tasks
    {
        "task": "Generate embeddings for product descriptions using OpenAI's text-embedding-3-small model. The output should include a 'description_embedding' field (1536 dimensions) and 'embedding_model' field.",
        "input": {
            "product_id": "P001",
            "name": "Wireless Bluetooth Headphones",
            "description": "Premium over-ear headphones with active noise cancellation and 30-hour battery life"
        },
        "output": {
            "product_id": "P001",
            "name": "Wireless Bluetooth Headphones",
            "description": "Premium over-ear headphones with active noise cancellation and 30-hour battery life",
            "description_embedding": "[vector of 1536 dimensions]",
            "embedding_model": "text-embedding-3-small"
        },
        "ground_truth_code": """def transform(data):
    import openai
    
    result = data.copy()
    
    if 'description' in result and result['description']:
        response = openai.embeddings.create(
            input=result['description'],
            model="text-embedding-3-small"
        )
        result['description_embedding'] = response.data[0].embedding
        result['embedding_model'] = "text-embedding-3-small"
    
    return result"""
    },
    {
        "task": "Generate embeddings for customer reviews using Cohere embed-english-v3.0 model with input_type='search_document'. The output should include a 'review_embedding' field (1024 dimensions) and 'embedding_model' field.",
        "input": {
            "review_id": "R001",
            "customer_name": "John Doe",
            "review_text": "This product exceeded my expectations. Great quality and fast shipping!"
        },
        "output": {
            "review_id": "R001",
            "customer_name": "John Doe",
            "review_text": "This product exceeded my expectations. Great quality and fast shipping!",
            "review_embedding": "[vector of 1024 dimensions]",
            "embedding_model": "embed-english-v3.0"
        },
        "ground_truth_code": """def transform(data):
    import cohere
    
    result = data.copy()
    co = cohere.Client(api_key='YOUR_API_KEY')
    
    if 'review_text' in result and result['review_text']:
        response = co.embed(
            texts=[result['review_text']],
            model='embed-english-v3.0',
            input_type='search_document'
        )
        result['review_embedding'] = response.embeddings[0]
        result['embedding_model'] = "embed-english-v3.0"
    
    return result"""
    },
    {
        "task": "Generate embeddings for support tickets using HuggingFace sentence-transformers model 'all-MiniLM-L6-v2'. The output should include a 'description_embedding' field (384 dimensions) and 'embedding_model' field.",
        "input": {
            "ticket_id": "T001",
            "subject": "Login Issues",
            "description": "Unable to login to my account. Getting error message: Invalid credentials"
        },
        "output": {
            "ticket_id": "T001",
            "subject": "Login Issues",
            "description": "Unable to login to my account. Getting error message: Invalid credentials",
            "description_embedding": "[vector of 384 dimensions]",
            "embedding_model": "all-MiniLM-L6-v2"
        },
        "ground_truth_code": """def transform(data):
    from sentence_transformers import SentenceTransformer
    
    result = data.copy()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    if 'description' in result and result['description']:
        embedding = model.encode(result['description'])
        result['description_embedding'] = embedding.tolist()
        result['embedding_model'] = "all-MiniLM-L6-v2"
    
    return result"""
    },
    {
        "task": "Combine multiple text fields (title, abstract, keywords) with '. ' separator and generate embeddings using OpenAI's text-embedding-3-small model. The output should include 'combined_text', 'combined_embedding' (1536 dimensions), and 'embedding_model' fields.",
        "input": {
            "article_id": "A001",
            "title": "The Future of AI in Healthcare",
            "abstract": "This article explores emerging AI technologies transforming medical diagnosis and treatment.",
            "keywords": "artificial intelligence, healthcare, medical diagnosis"
        },
        "output": {
            "article_id": "A001",
            "title": "The Future of AI in Healthcare",
            "abstract": "This article explores emerging AI technologies transforming medical diagnosis and treatment.",
            "keywords": "artificial intelligence, healthcare, medical diagnosis",
            "combined_text": "The Future of AI in Healthcare. This article explores emerging AI technologies transforming medical diagnosis and treatment. artificial intelligence, healthcare, medical diagnosis",
            "combined_embedding": "[vector of 1536 dimensions]",
            "embedding_model": "text-embedding-3-small"
        },
        "ground_truth_code": """def transform(data):
    import openai
    
    result = data.copy()
    
    # Combine multiple text fields
    text_parts = []
    for field in ['title', 'abstract', 'keywords']:
        if field in result and result[field]:
            text_parts.append(result[field])
    
    combined_text = '. '.join(text_parts)
    result['combined_text'] = combined_text
    
    # Generate embedding
    response = openai.embeddings.create(
        input=combined_text,
        model="text-embedding-3-small"
    )
    result['combined_embedding'] = response.data[0].embedding
    result['embedding_model'] = "text-embedding-3-small"
    
    return result"""
    },
    {
        "task": "Generate embeddings for job postings using Voyage AI voyage-2 model with input_type='document'. The output should include 'description_embedding' field (1024 dimensions) and 'embedding_model' field.",
        "input": {
            "job_id": "J001",
            "title": "Senior Data Engineer",
            "description": "We are seeking an experienced data engineer to build and maintain our data infrastructure. Must have 5+ years of experience with Python, SQL, and cloud platforms."
        },
        "output": {
            "job_id": "J001",
            "title": "Senior Data Engineer",
            "description": "We are seeking an experienced data engineer to build and maintain our data infrastructure. Must have 5+ years of experience with Python, SQL, and cloud platforms.",
            "description_embedding": "[vector of 1024 dimensions]",
            "embedding_model": "voyage-2"
        },
        "ground_truth_code": """def transform(data):
    import voyageai
    
    result = data.copy()
    vo = voyageai.Client(api_key='YOUR_API_KEY')
    
    if 'description' in result and result['description']:
        response = vo.embed(
            [result['description']],
            model="voyage-2",
            input_type="document"
        )
        result['description_embedding'] = response.embeddings[0]
        result['embedding_model'] = "voyage-2"
    
    return result"""
    },
    {
        "task": "Generate embeddings for blog posts using Jina AI jina-embeddings-v2-base-en model via API. The output should include 'content_embedding' field (768 dimensions) and 'embedding_model' field.",
        "input": {
            "post_id": "B001",
            "title": "10 Tips for Better Python Code",
            "content": "Writing clean, maintainable Python code is essential for any developer. In this post, we'll explore best practices including proper naming conventions, effective use of list comprehensions, and the importance of documentation."
        },
        "output": {
            "post_id": "B001",
            "title": "10 Tips for Better Python Code",
            "content": "Writing clean, maintainable Python code is essential for any developer. In this post, we'll explore best practices including proper naming conventions, effective use of list comprehensions, and the importance of documentation.",
            "content_embedding": "[vector of 768 dimensions]",
            "embedding_model": "jina-embeddings-v2-base-en"
        },
        "ground_truth_code": """def transform(data):
    import requests
    
    result = data.copy()
    
    if 'content' in result and result['content']:
        response = requests.post(
            'https://api.jina.ai/v1/embeddings',
            headers={
                'Content-Type': 'application/json',
                'Authorization': 'Bearer YOUR_API_KEY'
            },
            json={
                'input': [result['content']],
                'model': 'jina-embeddings-v2-base-en'
            }
        )
        embedding_data = response.json()
        result['content_embedding'] = embedding_data['data'][0]['embedding']
        result['embedding_model'] = "jina-embeddings-v2-base-en"
    
    return result"""
    },
    {
        "task": "Generate embeddings for multi-lingual content using Jina AI jina-embeddings-v2-base-multilingual model via API. The output should include 'text_embedding' field (768 dimensions) and 'embedding_model' field.",
        "input": {
            "message_id": "M002",
            "language": "es",
            "text": "Hola, estoy interesado en obtener más información sobre sus productos y servicios. ¿Podrían enviarme un catálogo?"
        },
        "output": {
            "message_id": "M002",
            "language": "es",
            "text": "Hola, estoy interesado en obtener más información sobre sus productos y servicios. ¿Podrían enviarme un catálogo?",
            "text_embedding": "[vector of 768 dimensions]",
            "embedding_model": "jina-embeddings-v2-base-multilingual"
        },
        "ground_truth_code": """def transform(data):
    import requests
    
    result = data.copy()
    
    if 'text' in result and result['text']:
        response = requests.post(
            'https://api.jina.ai/v1/embeddings',
            headers={
                'Content-Type': 'application/json',
                'Authorization': 'Bearer YOUR_API_KEY'
            },
            json={
                'input': [result['text']],
                'model': 'jina-embeddings-v2-base-multilingual'
            }
        )
        embedding_data = response.json()
        result['text_embedding'] = embedding_data['data'][0]['embedding']
        result['embedding_model'] = "jina-embeddings-v2-base-multilingual"
    
    return result"""
    },
    
    # LLM Summarization Tasks
    {
        "task": "Generate summary for customer feedback using OpenAI GPT-4.",
        "input": {
            "feedback_id": "F001",
            "customer_feedback": "I've been using this product for three months now and I'm really impressed with the build quality. The battery life is excellent, lasting me through two full days of heavy use. However, I did notice that the Bluetooth connection sometimes drops when I'm more than 20 feet away from my phone. The customer service team was very helpful when I reached out about this issue. Overall, I would definitely recommend this to anyone looking for a reliable device in this price range."
        },
        "output": {
            "feedback_id": "F001",
            "customer_feedback": "I've been using this product for three months now and I'm really impressed with the build quality. The battery life is excellent, lasting me through two full days of heavy use. However, I did notice that the Bluetooth connection sometimes drops when I'm more than 20 feet away from my phone. The customer service team was very helpful when I reached out about this issue. Overall, I would definitely recommend this to anyone looking for a reliable device in this price range.",
            "summary": "Customer praises excellent build quality and battery life but notes occasional Bluetooth connectivity issues beyond 20 feet. Positive experience with customer service. Recommends product overall."
        },
        "ground_truth_code": """def transform(data):
    import openai
    
    result = data.copy()
    
    if 'customer_feedback' in result and result['customer_feedback']:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes customer feedback concisely."},
                {"role": "user", "content": f"Summarize this customer feedback in 2-3 sentences: {result['customer_feedback']}"}
            ],
            temperature=0.3,
            max_tokens=150
        )
        result['summary'] = response.choices[0].message.content.strip()
    
    return result"""
    },
    {
        "task": "Generate summary for news articles using Anthropic Claude.",
        "input": {
            "article_id": "N001",
            "article_text": "Scientists at MIT have developed a new battery technology that could revolutionize electric vehicles. The breakthrough involves using solid-state electrolytes instead of liquid ones, which significantly improves energy density and safety. Early tests show the batteries can charge to 80% capacity in just 15 minutes and last for over 500,000 miles. The research team expects commercial production to begin within the next three years, pending further testing and regulatory approval."
        },
        "output": {
            "article_id": "N001",
            "article_text": "Scientists at MIT have developed a new battery technology that could revolutionize electric vehicles. The breakthrough involves using solid-state electrolytes instead of liquid ones, which significantly improves energy density and safety. Early tests show the batteries can charge to 80% capacity in just 15 minutes and last for over 500,000 miles. The research team expects commercial production to begin within the next three years, pending further testing and regulatory approval.",
            "summary": "MIT researchers have created a solid-state battery technology for EVs that charges faster and lasts longer than current options, with commercial production expected in three years."
        },
        "ground_truth_code": """def transform(data):
    import anthropic
    
    result = data.copy()
    client = anthropic.Anthropic(api_key='YOUR_API_KEY')
    
    if 'article_text' in result and result['article_text']:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=150,
            messages=[
                {"role": "user", "content": f"Summarize this article in one concise sentence: {result['article_text']}"}
            ]
        )
        result['summary'] = message.content[0].text.strip()
    
    return result"""
    },
    
    # LLM Classification Tasks
    {
        "task": "Classify sentiment of product reviews as positive, negative, or neutral using OpenAI.",
        "input": {
            "review_id": "R002",
            "review_text": "This product is absolutely terrible. It broke after just two weeks of normal use. Complete waste of money. Would not recommend to anyone."
        },
        "output": {
            "review_id": "R002",
            "review_text": "This product is absolutely terrible. It broke after just two weeks of normal use. Complete waste of money. Would not recommend to anyone.",
            "sentiment": "negative",
            "confidence": 0.95
        },
        "ground_truth_code": """def transform(data):
    import openai
    import json
    
    result = data.copy()
    
    if 'review_text' in result and result['review_text']:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis expert. Classify the sentiment as positive, negative, or neutral. Respond with JSON: {\"sentiment\": \"positive|negative|neutral\", \"confidence\": 0.0-1.0}"},
                {"role": "user", "content": result['review_text']}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        sentiment_data = json.loads(response.choices[0].message.content)
        result['sentiment'] = sentiment_data['sentiment']
        result['confidence'] = sentiment_data['confidence']
    
    return result"""
    },
    {
        "task": "Classify support ticket priority (low, medium, high, critical) using Claude.",
        "input": {
            "ticket_id": "T002",
            "subject": "System Down - Production Database Unreachable",
            "description": "Our entire production database has been unreachable for the past 15 minutes. All customer-facing services are down. This is affecting approximately 50,000 active users."
        },
        "output": {
            "ticket_id": "T002",
            "subject": "System Down - Production Database Unreachable",
            "description": "Our entire production database has been unreachable for the past 15 minutes. All customer-facing services are down. This is affecting approximately 50,000 active users.",
            "priority": "critical",
            "reasoning": "Complete production outage affecting large user base requires immediate attention"
        },
        "ground_truth_code": """def transform(data):
    import anthropic
    import json
    
    result = data.copy()
    client = anthropic.Anthropic(api_key='YOUR_API_KEY')
    
    if 'description' in result and result['description']:
        prompt = f\"\"\"Classify the priority of this support ticket as: low, medium, high, or critical.
        
Subject: {result.get('subject', '')}
Description: {result['description']}

Respond with JSON: {{"priority": "low|medium|high|critical", "reasoning": "brief explanation"}}\"\"\"
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract JSON from response
        response_text = message.content[0].text
        priority_data = json.loads(response_text)
        result['priority'] = priority_data['priority']
        result['reasoning'] = priority_data['reasoning']
    
    return result"""
    },
    {
        "task": "Classify customer intent from support messages (billing, technical, general_inquiry, complaint).",
        "input": {
            "message_id": "M001",
            "message_text": "Hi, I noticed an extra charge of $29.99 on my last invoice. I don't recall authorizing this. Can you help me understand what this charge is for?"
        },
        "output": {
            "message_id": "M001",
            "message_text": "Hi, I noticed an extra charge of $29.99 on my last invoice. I don't recall authorizing this. Can you help me understand what this charge is for?",
            "intent": "billing",
            "sub_intent": "disputed_charge"
        },
        "ground_truth_code": """def transform(data):
    import openai
    import json
    
    result = data.copy()
    
    if 'message_text' in result and result['message_text']:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Classify customer intent: billing, technical, general_inquiry, or complaint. Also provide sub_intent for more specific categorization. Respond with JSON."},
                {"role": "user", "content": result['message_text']}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        intent_data = json.loads(response.choices[0].message.content)
        result['intent'] = intent_data.get('intent')
        result['sub_intent'] = intent_data.get('sub_intent')
    
    return result"""
    },
    {
        "task": "Classify content moderation categories (safe, spam, harassment, hate_speech, violence).",
        "input": {
            "content_id": "C001",
            "user_content": "Check out this amazing deal! Click here now to claim your FREE iPhone! Limited time offer! bit.ly/notascam"
        },
        "output": {
            "content_id": "C001",
            "user_content": "Check out this amazing deal! Click here now to claim your FREE iPhone! Limited time offer! bit.ly/notascam",
            "moderation_category": "spam",
            "action_required": "remove",
            "confidence": 0.92
        },
        "ground_truth_code": """def transform(data):
    import openai
    import json
    
    result = data.copy()
    
    if 'user_content' in result and result['user_content']:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a content moderation system. Classify content as: safe, spam, harassment, hate_speech, or violence. Provide action_required: none, flag, or remove. Include confidence score. Respond with JSON."},
                {"role": "user", "content": result['user_content']}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        moderation_data = json.loads(response.choices[0].message.content)
        result['moderation_category'] = moderation_data['moderation_category']
        result['action_required'] = moderation_data['action_required']
        result['confidence'] = moderation_data['confidence']
    
    return result"""
    },
    
    # Combined Embedding + LLM Tasks
    {
        "task": "Generate embeddings using OpenAI text-embedding-3-small (1536 dimensions) and classify industry sector from company descriptions using GPT-4. Output should include 'description_embedding', 'industry_sector', and 'sub_sectors' fields.",
        "input": {
            "company_id": "CO001",
            "company_description": "We develop cloud-based software solutions for healthcare providers, including electronic health records, patient scheduling, and telemedicine platforms."
        },
        "output": {
            "company_id": "CO001",
            "company_description": "We develop cloud-based software solutions for healthcare providers, including electronic health records, patient scheduling, and telemedicine platforms.",
            "description_embedding": "[vector of 1536 dimensions]",
            "industry_sector": "healthcare_technology",
            "sub_sectors": ["healthcare", "saas", "telemedicine"]
        },
        "ground_truth_code": """def transform(data):
    import openai
    import json
    
    result = data.copy()
    
    if 'company_description' in result and result['company_description']:
        # Generate embedding
        embed_response = openai.embeddings.create(
            input=result['company_description'],
            model="text-embedding-3-small"
        )
        result['description_embedding'] = embed_response.data[0].embedding
        
        # Classify industry
        classify_response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Classify the company's primary industry sector and list relevant sub-sectors. Respond with JSON: {\"industry_sector\": \"sector_name\", \"sub_sectors\": [\"list\", \"of\", \"subsectors\"]}"},
                {"role": "user", "content": result['company_description']}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        classification = json.loads(classify_response.choices[0].message.content)
        result['industry_sector'] = classification['industry_sector']
        result['sub_sectors'] = classification['sub_sectors']
    
    return result"""
    },
    {
        "task": "Use GPT-4 to generate summary and classify research topics, then generate embeddings using OpenAI text-embedding-3-small (1536 dimensions). Output should include 'summary', 'research_topics', and 'abstract_embedding' fields.",
        "input": {
            "paper_id": "P001",
            "title": "Attention Is All You Need",
            "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."
        },
        "output": {
            "paper_id": "P001",
            "title": "Attention Is All You Need",
            "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
            "summary": "Introduces the Transformer architecture, a novel neural network based purely on attention mechanisms, eliminating the need for recurrent or convolutional layers.",
            "research_topics": ["deep_learning", "natural_language_processing", "neural_architecture"],
            "abstract_embedding": "[vector of 1536 dimensions]"
        },
        "ground_truth_code": """def transform(data):
    import openai
    import json
    
    result = data.copy()
    
    if 'abstract' in result and result['abstract']:
        # Generate summary
        summary_response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Summarize research paper abstracts in one clear sentence."},
                {"role": "user", "content": result['abstract']}
            ],
            temperature=0.3,
            max_tokens=100
        )
        result['summary'] = summary_response.choices[0].message.content.strip()
        
        # Classify topics
        topic_response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Identify the main research topics from this abstract. Respond with JSON: {\"research_topics\": [\"topic1\", \"topic2\", ...]}"},
                {"role": "user", "content": result['abstract']}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        topics = json.loads(topic_response.choices[0].message.content)
        result['research_topics'] = topics['research_topics']
        
        # Generate embedding
        embed_response = openai.embeddings.create(
            input=result['abstract'],
            model="text-embedding-3-small"
        )
        result['abstract_embedding'] = embed_response.data[0].embedding
    
    return result"""
    },
    {
        "task": "Use GPT-4 to extract key entities (parties, date, contract_type, duration, rate) and generate summary from legal documents, then create embeddings using OpenAI text-embedding-3-small (1536 dimensions). Output should include 'entities', 'summary', and 'document_embedding' fields.",
        "input": {
            "document_id": "D001",
            "document_text": "This Agreement is entered into as of January 15, 2024, by and between Acme Corporation, a Delaware corporation ('Company'), and John Smith, an individual residing in California ('Consultant'). The Consultant agrees to provide software development services for a period of 6 months at a rate of $150 per hour, not to exceed 40 hours per week. Payment shall be made monthly within 15 days of invoice receipt."
        },
        "output": {
            "document_id": "D001",
            "document_text": "This Agreement is entered into as of January 15, 2024, by and between Acme Corporation, a Delaware corporation ('Company'), and John Smith, an individual residing in California ('Consultant'). The Consultant agrees to provide software development services for a period of 6 months at a rate of $150 per hour, not to exceed 40 hours per week. Payment shall be made monthly within 15 days of invoice receipt.",
            "summary": "Service agreement between Acme Corporation and consultant John Smith for 6-month software development engagement at $150/hour with monthly payment terms.",
            "entities": {
                "parties": ["Acme Corporation", "John Smith"],
                "date": "2024-01-15",
                "contract_type": "consulting_agreement",
                "duration": "6 months",
                "rate": "$150/hour"
            },
            "document_embedding": "[vector of 1536 dimensions]"
        },
        "ground_truth_code": """def transform(data):
    import openai
    import json
    
    result = data.copy()
    
    if 'document_text' in result and result['document_text']:
        # Extract entities
        entity_response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Extract key entities from legal documents. Respond with JSON containing: parties, date, contract_type, duration, rate, and other relevant fields."},
                {"role": "user", "content": result['document_text']}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        result['entities'] = json.loads(entity_response.choices[0].message.content)
        
        # Generate summary
        summary_response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Summarize legal documents in one clear sentence."},
                {"role": "user", "content": result['document_text']}
            ],
            temperature=0.3,
            max_tokens=100
        )
        result['summary'] = summary_response.choices[0].message.content.strip()
        
        # Generate embedding
        embed_response = openai.embeddings.create(
            input=result['document_text'],
            model="text-embedding-3-small"
        )
        result['document_embedding'] = embed_response.data[0].embedding
    
    return result"""
    }
]

output_file = Path("etl_llm_embedding_dataset.json")
with open(output_file, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"LLM/Embedding ETL dataset with {len(dataset)} test cases written to {output_file}")
print(f"\nDataset breakdown:")
print(f"  - Embedding generation tasks: 7")
print(f"  - LLM summarization tasks: 2")
print(f"  - LLM classification tasks: 4")
print(f"  - Combined embedding + LLM tasks: 3")
print(f"\nAPIs covered:")
print(f"  - OpenAI (GPT-4, text-embedding-3-small)")
print(f"  - Anthropic (Claude)")
print(f"  - Cohere (embed-english-v3.0)")
print(f"  - HuggingFace (sentence-transformers)")
print(f"  - Voyage AI (voyage-2)")
print(f"  - Jina AI (jina-embeddings-v2-base-en, jina-embeddings-v2-base-multilingual)")
print(f"\nClassification types:")
print(f"  - Sentiment analysis")
print(f"  - Priority classification")
print(f"  - Intent detection")
print(f"  - Content moderation")
print(f"  - Industry/topic classification")