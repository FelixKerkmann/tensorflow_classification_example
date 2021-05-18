import shutil
import os
import openpyxl
import pdf2image
import numpy as np
import PIL

if __name__ == '__main__':

    xlsx_path = 'D:/IP5/Dokument_Metadaten.xlsx'
    dst_dir = 'D:/IP5/USB/pictures/'

    xlsx = openpyxl.load_workbook(xlsx_path)
    sheet = xlsx.active
    dim = sheet.dimensions
    rows = sheet.rows

    for row in rows:
        # print([str('D'+row)], " ", [str(('J'+row))])
        if row[3].value == 'UB':
            pass
        else:
            cat = row[3].value
            src = row[9].value
            dst = ''
            flag = -1
            for i in range(len(src)):
                if src[i] != '\\':
                    continue
                flag = i
            if cat == '213 Externer Bericht':
                dst = dst_dir + '213_Externer_Bericht/' + src[flag+1:]
            elif cat == '207 Einverständniserklärung':
                dst = dst_dir + '207_Einverständniserklärung/' + src[flag+1:]
            elif cat == '02 Zuweisung':
                dst = dst_dir + '02_Zuweisung/' + src[flag+1:]
            elif cat == '227 Pflegeprotokolle':
                dst = dst_dir + '227_Pflegeprotokolle/' + src[flag+1:]
            elif cat == '214 Externes EKG':
                dst = dst_dir + '214_Externes_EKG/' + src[flag+1:]
            else:
                dst = dst_dir + '0_Andere/' + src[flag+1:]
            # shutil.copy(src, dst)
            # return
            images = pdf2image.convert_from_path(src)
            images[0].save(str(dst[:-3] + 'png'))
            # print(src, str(dst[:-3] + 'png'))
