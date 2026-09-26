"""Golden Set: 5 clientes fictícios, um por segmento relevante."""

GOLDEN_SET = [
    {"id": 1, "descricao": "Jovem de 25 anos, nunca participou de campanha",
     "age": 25, "poutcome": "nonexistent", "segment": "jovem_sem_sucesso",
     "expected_channel": "cellular",
     "justificativa": "No histórico do segmento o celular converte bem mais que o telefone."},
    {"id": 2, "descricao": "Adulto de 45 anos, nunca participou de campanha",
     "age": 45, "poutcome": "nonexistent", "segment": "adulto_sem_sucesso",
     "expected_channel": "cellular",
     "justificativa": "Maior segmento da base; celular com conversão consistentemente maior."},
    {"id": 3, "descricao": "Adulto de 40 anos que aceitou a campanha anterior",
     "age": 40, "poutcome": "success", "segment": "adulto_com_sucesso",
     "expected_channel": "cellular",
     "justificativa": "Conversão alta nos dois canais, com vantagem do celular."},
    {"id": 4, "descricao": "Sênior de 70 anos cuja campanha anterior falhou",
     "age": 70, "poutcome": "failure", "segment": "senior_sem_sucesso",
     "expected_channel": "cellular",
     "justificativa": "Celular com conversão maior no segmento sênior sem sucesso anterior."},
    {"id": 5, "descricao": "Sênior de 65 anos que aceitou a campanha anterior",
     "age": 65, "poutcome": "success", "segment": "senior_com_sucesso",
     "expected_channel": None,
     "justificativa": ("Telefone 84,6% (n=13) vs. celular 75,4% (n=191): amostra pequena. "
                       "Espera-se baixa confiança: 0,05 < P(celular melhor) < 0,95.")},
]
