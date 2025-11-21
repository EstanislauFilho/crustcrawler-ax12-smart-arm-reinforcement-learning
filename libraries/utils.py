
import pickle
from datetime import datetime

# **********************************************************************

def convert_time(seconds): 
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)

# **********************************************************************

def execution_date():
    now = datetime.now() 
    return now.strftime("dt_%d-%m-%Y_hr_%H-%M-%S")

# **********************************************************************

def ler_arquivo_pose(caminho_txt):
    with open(caminho_txt, 'r') as f:
        conteudo = f.read()

    linhas = re.findall(r'(\w+)\s*:\s*([-+]?\d*\.?\d+)', conteudo)

    # Converte para dicionário com float
    dados = {chave: float(valor) for chave, valor in linhas}

    return dados