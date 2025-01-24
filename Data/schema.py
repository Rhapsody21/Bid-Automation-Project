import weaviate

# Connect to the Weaviate instance using the updated client
client = weaviate.WeaviateClient("http://localhost:8080")

# Define the schema
schema = {
    "classes": [
        {
            "class": "TechnicalProposal",
            "description": "Stores technical proposal documents",
            "properties": [
                {
                    "name": "title",
                    "dataType": ["text"],
                    "description": "The title of the technical proposal"
                },
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "The full content of the technical proposal"
                },
                {
                    "name": "metadata",
                    "dataType": ["text"],  # Use text for JSON-like metadata
                    "description": "Metadata in JSON format about the technical proposal"
                }
            ]
        },
        {
            "class": "FinancialProposal",
            "description": "Stores financial proposal documents",
            "properties": [
                {
                    "name": "title",
                    "dataType": ["text"],
                    "description": "The title of the financial proposal"
                },
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "The full content of the financial proposal"
                },
                {
                    "name": "metadata",
                    "dataType": ["text"],  # Use text for JSON-like metadata
                    "description": "Metadata in JSON format about the financial proposal"
                }
            ]
        },
        {
            "class": "RequestForProposal",
            "description": "Stores RFP documents",
            "properties": [
                {
                    "name": "title",
                    "dataType": ["text"],
                    "description": "The title of the RFP"
                },
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "The full content of the RFP"
                },
                {
                    "name": "metadata",
                    "dataType": ["text"],  # Use text for JSON-like metadata
                    "description": "Metadata in JSON format about the RFP"
                }
            ]
        }
    ]
}

# Create the schema in Weaviate
client.schema.create(schema)
