# Import drain3 model
from drain3.template_miner import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.persistence_handler import PersistenceHandler

def Drain_Init() -> TemplateMiner:
    """
    Create drain3 model
    
    This function will init a drain cluster model. This  model will be used for all drain log parsing progress.
    """
    ph = PersistenceHandler()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    drain3Config = TemplateMinerConfig()
    drain3Config.load('./config/drain3.ini')
    drain3Config.profiling_enabled = True

    tmp = TemplateMiner(config=drain3Config, persistence_handler=ph)
    return tmp
  
def RemoveSignals(line: str):
  """
  Remove all signals, numbers and single alpha from log line
  
  This function will remove the meaningless signals and numbers from the str.
  
  Parameter:
    line: The string value which is need to remove signals.
  """
  remove_list = list("~`!@#$%^&*()-_=+[{]};:'\",<.>/?|\\0123456789")
  res = line
  for signal in remove_list:
      res = res.replace(signal, ' ')
  res_list = res.split()
  alpha = "abcdefghijklmnopqrstuvwxyz"
  alpha_upper = alpha.upper()
  alpha_lower = alpha.lower()
  alpha_list = list(alpha_upper + alpha_lower)
  for a in alpha_list:
      while a in res_list:
          res_list.remove(a)
  res = ' '.join(res_list)
  return res
