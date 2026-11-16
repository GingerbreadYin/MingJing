"""Local operator entry point; credentials are consumed only by application code."""
from pathlib import Path
import os, sys


class RedactedStream:
    def __init__(self, stream):
        self.stream=stream
    def write(self, text):
        from .permissions import redact_runtime_secrets
        secret=os.environ.get('DEEPSEEK_API_KEY')
        if secret and secret != 'not-configured':
            text = text.replace(secret, '[REDACTED]')
        return self.stream.write(redact_runtime_secrets(text))
    def flush(self):
        return self.stream.flush()
    def __getattr__(self,name):
        return getattr(self.stream,name)


def prepare_environment():
    root=Path(__file__).resolve().parents[1]
    # The native MCP client must reach the local Gateway even on proxy-managed hosts.
    bypass = os.environ.get('no_proxy') or os.environ.get('NO_PROXY', '')
    os.environ['NO_PROXY'] = ','.join(dict.fromkeys([*filter(None, bypass.split(',')), '127.0.0.1', 'localhost', '::1']))
    os.environ['no_proxy'] = os.environ['NO_PROXY']
    os.environ.update(RAGOPT_ROOT=str(root),DEER_FLOW_PROJECT_ROOT=str(root),DEER_FLOW_HOME=str(root/'runtime/deerflow'),
                      DEER_FLOW_CONFIG_PATH=str(root/'config.yaml'),DEER_FLOW_EXTENSIONS_CONFIG_PATH=str(root/'extensions_config.json'),
                      HF_HOME=str(root/'.cache/models'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONIOENCODING='utf-8',
                      DEER_FLOW_AUTH_DISABLED='1',DEER_FLOW_ENV='development',LANGSMITH_TRACING='false',
                      LANGCHAIN_TRACING_V2='false',LANGFUSE_TRACING='false',MONOCLE_TRACING='false')
    from dotenv import load_dotenv
    selected=os.environ.get('DEER_FLOW_ENV_FILE')
    local=root/'.env'
    if selected:
        load_dotenv(selected,override=False)
    elif local.exists():
        os.environ['DEER_FLOW_ENV_FILE']=str(local)
        load_dotenv(local,override=False)
    # Allows read-only UI and offline checks without transmitting any request.
    os.environ.setdefault('DEEPSEEK_API_KEY','not-configured')
    if not isinstance(sys.stderr,RedactedStream):
        sys.stderr=RedactedStream(sys.stderr)
        sys.stdout=RedactedStream(sys.stdout)
    sys.path.insert(0,str(root/'upstream/backend'))
    return root


if __name__=='__main__':
    prepare_environment()
    import uvicorn
    uvicorn.run('app.gateway.app:app',host='127.0.0.1',port=8026,log_level='info',workers=1)
