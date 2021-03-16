import os
import numpy as np
from collections import OrderedDict 

def gen_flow_initial_test_mask_list(flow_root, output_txt_path):
    output_txt = open(output_txt_path, 'w')

    flow_list = [int(x[0:6]) for x in os.listdir(flow_root) if '.flo' in x]
    flow_list.sort()
    #print(flow_list)
    flow_list2 = [int(x[0:6]) for x in os.listdir(flow_root) if '.rflo' in x]
    flow_list2.sort()
    #print(flow_list2)
    #flow_list.extend(flow_list2)
    flow_num = len(flow_list) // 2
    flow_start_no = 0
    video_num = 0
    order_dict = {}
    for i in range(len(flow_list)):
        if(flow_list[i]%10 == 0):
          z = flow_list[i]//10 - 2
          if(z in order_dict.keys()):
            order_dict[z].append(flow_list[i])
          else:
            order_dict[z] = [flow_list[i]]
        elif(flow_list[i]%10 == 1):
          z = flow_list[i]//10 - 1
          if(z in order_dict.keys()):
            order_dict[z].append(flow_list[i])
          else:
            order_dict[z] = [flow_list[i]]
        elif(flow_list[i]%10 == 2):
          z = flow_list[i]//10 + 1
          if(z in order_dict.keys()):
            order_dict[z].append(flow_list[i])
          else:
            order_dict[z] = [flow_list[i]]
        elif(flow_list[i]%10 == 3):
          z = flow_list[i]//10 + 2
          if(z in order_dict.keys()):
            order_dict[z].append(flow_list[i])
          else:
            order_dict[z] = [flow_list[i]]
    order_dict = OrderedDict(sorted(order_dict.items())) 
    print(order_dict)
    order= sum(order_dict.values(),[])
    temp = order[0]
    temp2 = order[-1]
    order.insert(0,temp)
    order.insert(0,temp)
    order.insert(0,temp)
    order.insert(0,temp)
    order.insert(0,temp)
    order.insert(-1,temp2)
    order.insert(-1,temp2)
    order.insert(-1,temp2)
    order.insert(-1,temp2)
    order.insert(-1,temp2)
    print(order)
    list_minus = [-2,-1,1,2]
    for i in range(flow_start_no, len(order) - 10):
        for k in range(11):
            flow_no = order[i+k]
            output_txt.write('%06d.flo' % flow_no)
            output_txt.write(' ')
        for k in range(11):
            flow_no = order[i+k]//10
            if((flow_no + list_minus[order[i+k]%10] < 0) or (flow_no + list_minus[order[i+k]%10] > (temp2//10))):
              output_txt.write('%05d.png' % (flow_no))
            else:
              output_txt.write('%05d.png' % (flow_no + list_minus[order[i+k]%10]))
            output_txt.write(' ')
        output_txt.write('%06d.flo' % (order[i+5]))
        output_txt.write(' ')
        output_txt.write(str(video_num))
        output_txt.write('\n')
    for i in range(flow_start_no, len(order) - 10):
        for k in range(11):
            flow_no = order[i+k]
            output_txt.write('%06d.rflo' % flow_no)
            output_txt.write(' ')
        for k in range(11):
            flow_no = order[i+k]//10
            if((flow_no + list_minus[order[i+k]%10] < 0) or (flow_no + list_minus[order[i+k]%10] > (temp2//10))):
              output_txt.write('%05d.png' % (flow_no))
            else:
              output_txt.write('%05d.png' % (flow_no + list_minus[order[i+k]%10]))
            output_txt.write(' ')
        output_txt.write('%06d.rflo' % (order[i+5]))
        output_txt.write(' ')
        output_txt.write(str(video_num))
        output_txt.write('\n')

    output_txt.close()

def gen_flow_refine_test_mask_list(flow_root, output_txt_path):
    output_txt = open(output_txt_path, 'w')

    flow_list = [int(x[0:6]) for x in os.listdir(flow_root) if '.flo' in x]
    flow_list.sort()
    #print(flow_list)
    flow_list2 = [int(x[0:6]) for x in os.listdir(flow_root) if '.rflo' in x]
    flow_list2.sort()
    #print(flow_list2)
    #flow_list.extend(flow_list2)
    flow_num = len(flow_list) // 2
    flow_start_no = 0
    video_num = 0
    order_dict = {}
    for i in range(len(flow_list)):
        if(flow_list[i]%10 == 0):
          z = flow_list[i]//10 - 2
          if(z in order_dict.keys()):
            order_dict[z].append(flow_list[i])
          else:
            order_dict[z] = [flow_list[i]]
        elif(flow_list[i]%10 == 1):
          z = flow_list[i]//10 - 1
          if(z in order_dict.keys()):
            order_dict[z].append(flow_list[i])
          else:
            order_dict[z] = [flow_list[i]]
        elif(flow_list[i]%10 == 2):
          z = flow_list[i]//10 + 1
          if(z in order_dict.keys()):
            order_dict[z].append(flow_list[i])
          else:
            order_dict[z] = [flow_list[i]]
        elif(flow_list[i]%10 == 3):
          z = flow_list[i]//10 + 2
          if(z in order_dict.keys()):
            order_dict[z].append(flow_list[i])
          else:
            order_dict[z] = [flow_list[i]]
    order_dict = OrderedDict(sorted(order_dict.items())) 
    print(order_dict)
    order= sum(order_dict.values(),[])
    temp = order[0]
    temp2 = order[-1]
    order.insert(0,temp)
    order.insert(0,temp)
    order.insert(0,temp)
    order.insert(0,temp)
    order.insert(0,temp)
    order.insert(-1,temp2)
    order.insert(-1,temp2)
    order.insert(-1,temp2)
    order.insert(-1,temp2)
    order.insert(-1,temp2)
    print(order)
    list_minus = [-2,-1,1,2]
    for i in range(flow_start_no, len(order) - 10):
        for k in range(11):
            flow_no = order[i+k]
            output_txt.write('%06d.flo' % flow_no)
            output_txt.write(' ')
        for k in range(11):
            flow_no = order[i+k]
            output_txt.write('%06d.rflo' % flow_no)
            output_txt.write(' ')
        for k in range(11):
            flow_no = order[i+k]//10
            if((flow_no + list_minus[order[i+k]%10] < 0) or (flow_no + list_minus[order[i+k]%10] > (temp2//10))):
              output_txt.write('%05d.png' % (flow_no))
            else:
              output_txt.write('%05d.png' % (flow_no + list_minus[order[i+k]%10]))
            output_txt.write(' ')
        for k in range(11):
            flow_no = order[i+k]//10
            if((flow_no + list_minus[order[i+k]%10] < 0) or (flow_no + list_minus[order[i+k]%10] > (temp2//10))):
              output_txt.write('%05d.png' % (flow_no))
            else:
              output_txt.write('%05d.png' % (flow_no + list_minus[order[i+k]%10]))
            output_txt.write(' ')
        output_txt.write('%06d.flo' % (order[i+5]))
        output_txt.write(',')
        output_txt.write('%06d.rflo' % (order[i+5]))
        output_txt.write(' ')
        output_txt.write(str(video_num))
        output_txt.write('\n')

    output_txt.close()

# if __name__ == '__main__':
#     gen_flow_initial_test_mask_list(flow_root='', output_txt_path='')