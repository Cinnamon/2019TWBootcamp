import pandas as pd
import re

class invoice_compare():
    def __init__(self):
        global prize_num
        prize_num = pd.read_csv('invoice.csv', dtype=str)
        for i in range(len(prize_num['award3'])):
            prize_num['award3'][i] = prize_num['award3'][i].replace(' ', '')
            prize_num['award3'][i] = prize_num['award3'][i].split('、')
        for i in range(len(prize_num['award6'])):
            prize_num['award6'][i] = prize_num['award6'][i].replace(' ', '')
            prize_num['award6'][i] = prize_num['award6'][i].split('、')
        global prize_num_ls
        prize_num_ls=[]
        for month in prize_num['date']:
            prize_num_ls.append('prize_dict_'+month)
        
        for i in range(len(prize_num_ls)):
            prize_num_ls[i] = dict(award1 = prize_num['award1'][i], award2 = prize_num['award2'][i], award3 = prize_num['award3'][i],
                                 award6 = prize_num['award6'][i])
    @staticmethod
    def compare(num,month):
        global prize_num_ls
        prize_ls=[]
        prize_dict = prize_num_ls[month]
        if num == prize_dict['award1'].replace(" ", ""):
            prize_ls.append('特別獎')
        elif num == prize_dict['award2'].replace(" ", ""):
            prize_ls.append('特獎')
        for award3_num in prize_dict['award3']:
            if num == award3_num:
                prize_ls.append('特獎')
            elif num[-7:] == award3_num[-7:]:
                prize_ls.append('二獎')
            elif num[-6:] == award3_num[-6:]:
                prize_ls.append('三獎')
            elif num[-5:] == award3_num[-5:]:
                prize_ls.append('四獎')
            elif num[-4:] == award3_num[-4:]:
                prize_ls.append('五獎')
            elif num[-3:] == award3_num[-3:]:
                prize_ls.append('六獎')
        for award6_num in prize_dict['award6']:
            if num[-3:] == award6_num:
                prize_ls.append('六獎')
        else:
            prize_ls.append('沒中獎')
        return prize_ls[0]

    @staticmethod
    def num_filter(ocr_result):
        #input the outcome of OCR
        print("ocr_result: ",ocr_result)
        #print("type ocr_result: ",type(ocr_result))
        pattern = re.compile(r'\d{8}')
        res = re.findall(pattern, ocr_result)
        #print(res)
        if (len(res) == 0):
            return "No Number Found QAQ"
        else:
            return res[0][-8:]


    def get_table_month(self,month):
        global prize_num
        prize_num_in = prize_num
        return prize_num_in['date'][month]

    def get_month_info(self,month):
        global prize_num
        prize_num_in = prize_num
        info = prize_num_in['award1'][month], prize_num_in['award2'][month], prize_num_in['award3'][month], prize_num_in['award6'][month]
        return info
        
'''
model = invoice_compare()
pred_num = model.num_filter('sasa 00012136 sasa sqwq')
print('pred_num: ',pred_num)
month = 0#The newest
print(model.compare(pred_num,month))
'''
#keep show 沒中獎