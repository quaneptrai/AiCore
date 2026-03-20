from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_3CPrZH_65UmbWfwGXiJYiFqzciZudoXx57Au2F1jNysFMvTAj5tZqbQriUtR7o8wRSGTda")

# Lấy danh sách index
indexes = pc.list_indexes()

print("Indexes found:")
for idx in indexes:
    print("-", idx["name"])