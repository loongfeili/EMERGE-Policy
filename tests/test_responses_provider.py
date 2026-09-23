import asyncio
import json
import pytest
from Emerge.providers.responses_provider import ResponsesProvider
from Emerge.providers.openai_codex_provider import _convert_messages

class Stream:
    def __init__(self, events): self.events = events
    async def aiter_lines(self):
        for event in self.events:
            yield 'data: '+json.dumps(event)
            yield ''

def test_tool_result_ids_and_image_conversion():
    result=ResponsesProvider._parse({'status':'completed','output':[{'type':'function_call','call_id':'call_a','name':'look','arguments':'{"camera":"head"}'}]})
    _,items=_convert_messages([{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,YQ=='}}]}, {'role':'assistant','tool_calls':[result.tool_calls[0].to_openai_tool_call()]}, {'role':'tool','tool_call_id':'call_a','content':'visible'}])
    assert items[0]['content'][0]['type']=='input_image'
    assert items[1]['call_id']==items[2]['call_id']=='call_a'

def test_truncated_stream_never_counts_as_success():
    provider=object.__new__(ResponsesProvider)
    with pytest.raises(RuntimeError,match='without a terminal response'):
        asyncio.run(provider._consume(Stream([{'type':'response.output_text.delta','delta':'partial'}])))

def test_terminal_usage_and_tool_arguments():
    provider=object.__new__(ResponsesProvider)
    event={'type':'response.completed','response':{'status':'completed','usage':{'input_tokens':12,'output_tokens':5,'total_tokens':17},'output':[{'type':'function_call','call_id':'c','name':'move','arguments':'{"x":2}'}]}}
    result=asyncio.run(provider._consume(Stream([event])))
    assert result.tool_calls[0].arguments=={'x':2}
    assert result.usage['total_tokens']==17
