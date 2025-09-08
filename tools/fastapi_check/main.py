from typing import Union
import numpy as np

from fastapi import FastAPI, Request, Response, Depends
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}


async def parse_body(request: Request):
    data: bytes = await request.body()
    return data


@app.post("/foo")
async def parse_input(data: bytes = Depends(parse_body)):
    # Do something with data
    pass



def create_arr():
    w, h = 512, 512
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[0:256, 0:256] = [255, 0, 0] # red patch in upper left
    return arr


def read_arr_from_rawfile(path="$HOME/Downloads/arr", dtype_="uint8", shape_="512,512,3" ):
    """
    this suceeded to recover the numpy array from octet-stream of bytes download from /arr endpoint, 
    but it is cheating regards the metadata 
    """
    x = None

    dtype = getattr(np, dtype_, None)
    shape = tuple(map(int,shape_.split(","))) 

    with open(os.path.expandvars(path), "rb") as fp:
        xb = fp.read()
        x = np.frombuffer(xb, dtype=dtype ).reshape(*shape)
    pass
    return x



@app.get('/arr', response_class=Response)
def get_arr():
    """

    zeta:Downloads blyth$ curl -s -D /dev/stdout http://127.0.0.1:8000/arr  --output arr2
    HTTP/1.1 200 OK
    date: Mon, 08 Sep 2025 08:07:32 GMT
    server: uvicorn
    x-numpy-dtype: uint8
    x-numpy-shape: 512,512,3
    content-length: 786432
    content-type: application/octet-stream

    zeta:Downloads blyth$ echo $(( 512*512*3 ))
    786432

    """

    x = create_arr()
    x_bytes = x.tobytes('C')

    headers = {} 
    headers["x-numpy-dtype"] = x.dtype.name
    headers["x-numpy-shape"] = str(x.shape).replace(" ","")[1:-1]

    return Response(x_bytes, headers=headers, media_type='application/octet-stream')


async def parse_request_as_array(request: Request):
    """
    :param request: FastAPI Request
    :return arr: NumPy array 

    Uses request body and headers with array dtype and shape to reconstruct the uploaded NumPy array
    """
    data: bytes = await request.body()


    level_ = request.headers.get('x-numpy-level','0')
    dtype_ = request.headers.get('x-numpy-dtype')
    shape_ = request.headers.get('x-numpy-shape')

    level = int(level_)
    dtype = getattr(np, dtype_, None)
    shape = tuple(map(int,shape_.split(","))) 
    arr = np.frombuffer(data, dtype=dtype ).reshape(*shape)

    if level > 0:
        print("[parse_request_as_array")
        print(" level[%d]" % level )
        print(request.headers) 
        #print("data[%s]" % data )
        print("dtype_[%s]" % str(dtype) )
        print("shape_[%s]" % str(shape) )
        print("arr[%s]" % arr )
        print("]parse_request_as_array")
    pass

    return arr



def make_array_response( arr: np.array, media_type : str, level : int = 0 ):
    headers = {} 
    headers["x-numpy-level"] = str(level)
    headers["x-numpy-dtype"] = arr.dtype.name
    headers["x-numpy-shape"] = str(arr.shape).replace(" ","")[1:-1]
    return Response(arr.tobytes('C'), headers=headers, media_type=media_type )




@app.post('/upload_array', response_class=Response)
def upload_array(arr0: np.array = Depends(parse_request_as_array)):
    """
    :param arr0:
    :return response: Response

    1. parse_request_as_array providing the uploaded NumPy arr0
    2. operate on the uploaded array
    3. return the operated array as a FastAPI Response
    """

    arr = arr0 * 10.   # do some operation on the input array  

    response =  make_array_response(arr, "application/octet-stream" )

    return response 


@app.get("/legacy/")
def get_legacy_data():
    data = """<?xml version="1.0"?>
    <shampoo>
    <Header>
        Apply shampoo here.
    </Header>
    <Body>
        You'll have to use soap here.
    </Body>
    </shampoo>
    """
    return Response(content=data, media_type="application/xml")





