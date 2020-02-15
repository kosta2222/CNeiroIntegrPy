from struct import pack
push_i=0
push_fl=1
make_kernel=2
stop=b'\x03'
b_c=[]
def py_pack(op_i,val_i_or_fl):
    if op_i==push_fl:
        b_c.append(pack('B',push_fl))
        for i in pack('<f',val_i_or_fl):
            b_c.append(i.to_bytes(1,byteorder='little'))
    elif op_i==push_i:
        b_c.append(pack('B',push_i))
        b_c.append(pack('B',val_i_or_fl))
    elif op_i==make_kernel:
        b_c.append(pack('B',make_kernel))
def dump_bc(f_name):
    b_c.append(stop)
    with open(f_name,'wb') as f:
      for i in b_c:
        f.write(i)
