import numpy as np
import csv
import pandas as pd
import sys

def mutation_accessor(data):
    """Mutation Assessor[Uniprot-Accession,REF-AA,AA-POS,ALT-AA]"""
    """e.g.[EGFR-HUMAN R521K]"""
    index = []
    for j in range(len(data[0])):
        if data[0][j] == 'Uniprot-Accession':
            index.append(j)

    for j in range(len(data[0])):
        if data[0][j] == 'REF-AA':
            index.append(j)

    for j in range(len(data[0])):
        if data[0][j] == 'AA-Pos':
            index.append(j)

    for j in range(len(data[0])):
        if data[0][j] == 'ALT-AA':
            index.append(j)
    
    data = data[:,index]
    return data

def polyphen2(data):
    """PolyPhen-2[CHRL:Nuc-Pos,REF-Nuc/ALT-Nuc]"""
    """e.g.[chr1:1267483 G/A]"""
    index = []

    for j in range(len(data[0])):
        if data[0][j] == 'CHR':
            index.append(j)

    for j in range(len(data[0])):
        if data[0][j] == 'Nuc-Pos':
            index.append(j)

    for j in range(len(data[0])):
        if data[0][j] == 'REF-Nuc':
            index.append(j)

    for j in range(len(data[0])):
        if data[0][j] == 'ALT-Nuc':
            index.append(j)

    data = data[:,index] 

    return data

def sift(data):
    """SIFT[Ensembl-Protein-ID,AA-Pos,REF-AA,ALT-AA]"""
    """e.g.[ENSP00000322020,L150V]"""
    index = []
    for j in range(len(data[0])):
        if data[0][j] == 'Ensembl-Protein-ID':
            index.append(j)

    for j in range(len(data[0])):
        if data[0][j] == 'AA-Pos':
            index.append(j)

    for j in range(len(data[0])):
        if data[0][j] == 'REF-AA':
            index.append(j)

    for j in range(len(data[0])):
        if data[0][j] == 'ALT-AA':
            index.append(j)

    data = data[:,index]
    return data

def fathmmw(data):
    """SIFT[Ensembl-Protein-ID,AA-Pos,REF-AA,ALT-AA]"""
    """e.g.[ENSP00000322020 L150V]"""
    index = []
    for j in range(len(data[0])):
        if data[0][j] == 'Ensembl-Protein-ID':
            index.append(j)

    for j in range(len(data[0])):
        if data[0][j] == 'AA-Pos':
            index.append(j)

    for j in range(len(data[0])):
        if data[0][j] == 'REF-AA':
            index.append(j)

    for j in range(len(data[0])):
        if data[0][j] == 'ALT-AA':
            index.append(j)

    data = data[:,index]
    return data

def raw_data_process(file):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    data = np.array(data)
    

    data = fathmmw(data)

    # If the input contains chromosome position, the code is useful.
    # Otherwise, comment it.
    # delete = []
    # for i in range(len(data)):
    #     if i != 0:
    #         if data[i][0] == 'X':
    #             data[i][0] = 23
    #         if data[i][0] == 'Y':
    #             data[i][0] = 24
    #         if data[i][0] not in ['0','1','2','3','4','5','6','7','8','9' \
    #             ,'10','11','12','13','14','15','16','17','18','19' \
    #             ,'20','21','22','23','24']:
    #             delete.append(i)
    # data = np.delete(data,delete,0)
    
    return data

def write_csv_ma(data,file_name):
    file = open(file_name,'w')
    for i in range(len(data)):
        file.write(str(data[i][0]))
        file.write(" ")
        file.write(str(data[i][1]))
        file.write(str(data[i][2]))
        file.write(str(data[i][3]))
        file.write("\n")   

    file.close()

def write_csv_pp2(data,file_name):
    file = open(file_name,'w')
    for i in range(len(data)):
        file.write('chr')
        file.write(str(data[i][0]))
        file.write(":")
        file.write(str(data[i][1]))
        file.write(" ")
        file.write(str(data[i][2]))
        file.write("/")
        file.write(str(data[i][3]))
        file.write("\n")   

    file.close()

def write_csv_sift(data,file_name):
    file = open(file_name,'w')
    for i in range(len(data)):
        file.write(str(data[i][0]))
        file.write(",")
        file.write(str(data[i][1]))
        file.write(str(data[i][2]))
        file.write(str(data[i][3]))
        file.write("\n")   

    file.close()

def write_csv_fathmmw(data,file_name):
    file = open(file_name,'w')
    for i in range(len(data)):
        file.write(str(data[i][0]))
        file.write(" ")
        file.write(str(data[i][1]))
        file.write(str(data[i][2]))
        file.write(str(data[i][3]))
        file.write("\n")   

    file.close()


def write_csv(data,file_name):
    file = open(file_name,'w')
    for i in range(len(data)):
        file.write(str(data[i][0]))
        file.write(",")
        file.write(str(data[i][1]))
        file.write(",")
        file.write(str(data[i][2]))
        file.write(",")
        file.write(str(data[i][3]))
        file.write("\n")   

    file.close()


def read_sort_file(file):
    df = pd.DataFrame(pd.read_csv(file,header=0))
    # If the input contains chromosome position, the code is useful.
    # Otherwise, comment it.
    # df['CHR'] = df['CHR'].astype('int')
    # df = df.sort_values(by=['CHR','Nuc-Pos'],ascending=True)
    return df

if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file1 = sys.argv[2]
    output_file2 = sys.argv[3]

    data = raw_data_process(input_file)
    write_csv(data,output_file1)

    data = read_sort_file(output_file1)
    data = np.array(data)
    for i in range(len(data)):
        if data[i][0] == 23:
            data[i][0] = 'X'
        if data[i][0] == 24:
            data[i][0] = 'Y'
    write_csv_fathmmw(data,output_file2)