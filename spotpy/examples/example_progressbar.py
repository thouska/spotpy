import spotpy.tools.progressbar as pb

# Initial call to print 0% progress
pgr = 0
listlen = 1000000
pb.printProgressBar(pgr, listlen, prefix='Progress:', suffix='Complete', length=10)
for i in range(1000000):
    pgr += 1
    pb.printProgressBar(pgr, listlen, prefix='Progress:', suffix='Complete', length=10)

