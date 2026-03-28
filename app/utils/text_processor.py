import re

def super_clean(text: str):
    # 1. Remove quebras de linha que cortam palavras
    text = text.replace(' \n', ' ').replace('\n', ' ')
    # 2. Remove espaços duplos
    text = re.sub(r'\s+', ' ', text)
    # 3. Remove frases repetidas consecutivas (A gagueira do PDF)
    # Procura por padrões que se repetem como "Texto A Texto A"
    words = text.split()
    if not words: return ""
    
    cleaned_words = []
    i = 0
    while i < len(words):
        # Janela de comparação (tenta achar repetições de até 10 palavras)
        found_repeat = False
        for n in range(1, 11):
            if i + 2*n <= len(words):
                if words[i:i+n] == words[i+n:i+2*n]:
                    # Achou repetição, pula a primeira ocorrência
                    i += n
                    found_repeat = True
                    break
        if not found_repeat:
            cleaned_words.append(words[i])
            i += 1
    return " ".join(cleaned_words)