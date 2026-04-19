from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_4JWw7z_TY1XUsZDVAKwXpYEBiA1a9UCcraKD4MVmt9r9T56k1QNdzSo3Bgnep4bbgQTsW4")

index = pc.Index("thuctap")

index.delete(delete_all=True)