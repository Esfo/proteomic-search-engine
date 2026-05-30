from decimal import Decimal
from tempfile import NamedTemporaryFile
from time import time
import heapq
import csv
import os

def yieldlines(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        for row in file.readlines():
            yield row

def csvsplit(inputfile, chunksize=1000000):
    #chunk csv, sort, save temp files
    with open(inputfile, 'r', encoding='utf-8') as originalcsv:
        headers = originalcsv.readline()
        tempfiles = []
        while True:
            n = 0
            rows = []
            finished = False
            try:
                while True:
                    row = originalcsv.readline()
                    sequence, analyteid, score, ion_coverage, scan_indices = row.strip().split(',')
                    score = Decimal(score)
                    rows.append([score, row])
                    n += 1
                    if n > chunksize:
                        break
            except ValueError:
                #file is finished
                finished = True
            if rows:
                #sort by score
                rows = list(zip(*sorted(rows)))[1]
                #save temp file
                tempcsv = NamedTemporaryFile(mode='w', newline='', encoding='utf-8', delete=False)
                csvwriter = csv.writer(tempcsv)
                for row in rows:
                    csvwriter.writerow(row.strip().split(','))
                tempfiles.append(tempcsv.name)
                tempcsv.close()
            if finished:
                break
    return tempfiles, headers

def fileheaps(tempfiles, outputfile, headers):

    fileheap = [] #[[score, tempfile, row],...]
    fileyielders = {} #tempfile name: generator to yield its next row
    for t in tempfiles:
        #set up heaps of sorted files yielding lines
        fileyielders[t] = yieldlines(t)
        row = next(fileyielders[t])
        sequence, analyteid, score, ion_coverage, scan_indices = row.strip().split(',')
        score = Decimal(score)
        output = [score, t, row]
        heapq.heappush(fileheap, output)

    with open(outputfile, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers.strip().split(','))
        while fileheap:
            #write the latest sorted value to the csv
            score, t, row = heapq.heappop(fileheap)
            writer.writerow(row.strip().split(','))
            
            #keep track of the next latest line coming out of each file
            #yield the next value from the temp chunk that was just taken
            try:
                newrow = next(fileyielders[t])
            except StopIteration:
                #that file is done
                continue
            sequence, analyteid, score, ion_coverage, scan_indices = newrow.strip().split(',')
            score = Decimal(score)
            output = [score, t, newrow]
            heapq.heappush(fileheap, output)

def csv_score_sorting(processingdirectory):
    nt = time()
    peptidefilename = processingdirectory + 'peptiderankings.csv'
    tempfiles, headers = csvsplit(peptidefilename)
    fileheaps(tempfiles, peptidefilename, headers)

    for t in tempfiles:
        os.unlink(t)
    print(time() - nt, 'peptide rankings sorted')
