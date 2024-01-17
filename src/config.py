import yaml

class CfgDocument:
    path = None 

    def __init__(self, config):
        if config:
            self.path = config['document']['folder']

class CfgModel:
    m_llm = {}
    m_embedding = None

    def __init__(self, config):
        if config:
            self.m_llm['openai'] = self._get_model_llm(config, 'openai')
            self.m_llm['gpt4all'] = self._get_model_llm(config, 'gpt4all')
            self.m_embedding = self._get_model_embedding(config)

    def _get_model_llm(self, config, oper):
        key = config['model']['llm'][oper]['use']
        return config['model']['llm'][oper]['list'][key]
    
    def _get_model_embedding(self, config):
        key = config['model']['embedding']['use']
        return config['model']['embedding'][key]

class Config:
    config = None
    mode = None
    document: CfgDocument = None
    model: CfgModel = None

    def __init__(self, file):
        with open(file, "r") as f:
            self.config = yaml.safe_load(f)
        self.mode = self.config['mode']
        self._init_cfg_document()
        self._init_cfg_model()

    def _init_cfg_document(self):
        self.document = CfgDocument(config=self.config)

    def _init_cfg_model(self):
        self.model = CfgModel(config=self.config)

    def get(self, key):
        return self.config[key]