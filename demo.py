from src.pipeline import TurkishPreprocPipeline

p = TurkishPreprocPipeline()
res = p.process("Merhaba dünya! Nasılsın bugün? ")

print(res["tokens"])
print(res["sentences"])