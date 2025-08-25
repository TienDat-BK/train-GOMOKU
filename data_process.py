import os

def clean_data( folder_path):
    list_files = os.listdir(folder_path)
    # xử lí
    for file_path in list_files:
        f = open(os.path.join(folder_path,file_path), 'r+')
        list_of_line = f.readlines()

        if not any( char.isalpha() for char in list_of_line[0]):
            print('CLEANED')
            continue
        f.seek(0)
        f.truncate()

        for line in list_of_line[1:len(list_of_line) - 4]:
            x,y,z = line.strip().split(',') 
            new_line = x + ',' + y + '\n'
            f.writelines(new_line)   

        print(file_path + 'DONE!')



clean_data('dataTrain\\Freestyle15_1')
clean_data('dataTrain\\Freestyle15_2')

