
import os
# from dotenv import load_dotenv
from urllib.parse import urlencode, unquote, quote_plus
import requests
import json
from pprint import pprint

# load_dotenv(verbose=True)

end_poit = os.getenv('END_POINT')
key_decode = os.getenv('Dec_API_KEY')

decode_key = unquote(key_decode)

queryParams = '?' + urlencode({
    quote_plus('numOfRows') : '1',
    quote_plus('pageNo') : '1',
    quote_plus('resultType') : 'json',
    quote_plus('ServiceKey') : decode_key,
    quote_plus('basDt') : '202405020',
    # quote_plus('beginBasDt') : '20240101',
    # quote_plus('endBasDt') : '202405020',
})

#----------
response = requests.get(end_poit + queryParams)

response.encoding = 'utf-8'
print(response.status_code)
print(response.headers)
print(response.json())
# #----------
# if(rescode==200):
#     response_body = response.read()
#     dict_ = json.loads(response_body.decode('utf-8'))
#     pprint(dict_)
# else:
#     print("Error Code:" + rescode)      