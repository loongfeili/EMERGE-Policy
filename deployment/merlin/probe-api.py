import asyncio,json,time
from pathlib import Path
from Emerge.config.schema import Config
from Emerge.providers.factory import create_provider
async def main():
 cfg=Config.model_validate(json.loads(Path('/home/tiger/.config/emerge-robodojo/agent.json').read_text()))
 async def one(i):
  p=create_provider(cfg);start=time.monotonic()
  try:
   r=await p.chat_with_retry([{'role':'user','content':'Call check with value OK.'}],tools=[{'type':'function','function':{'name':'check','description':'Connectivity check','parameters':{'type':'object','properties':{'value':{'type':'string'}},'required':['value'],'additionalProperties':False}}}],max_tokens=128,reasoning_effort='low')
   return {'i':i,'ok':r.finish_reason=='tool_calls' and bool(r.tool_calls) and r.tool_calls[0].arguments=={'value':'OK'},'finish':r.finish_reason,'seconds':round(time.monotonic()-start,2)}
  except Exception as e:return {'i':i,'ok':False,'error_type':type(e).__name__}
  finally:await p.aclose()
 rows=await asyncio.gather(*(one(i) for i in range(1)));print(json.dumps({'concurrency':1,'passed':sum(r['ok'] for r in rows),'results':rows}))
asyncio.run(main())
