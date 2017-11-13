from math import sqrt
from PIL import Image, ImageDraw
import random

class bicluster:
	def __init__(self, vec, id=None, left=None, right=None, distance=0.0):
		self.id = id
		self.vec = vec
		self.left = left
		self.right = right
		self.distance = distance

def readfile(filename):
	lines = [line for line in file(filename)]

	# lines[0](first line) is the column names
	colnames = lines[0].strip().split('\t')[0:]

	rownames = []
	data = []

	for line in lines[1:]:
		p = line.strip().split('\t')
		#p[0] is the row name for all the rows
		rownames.append(p[0])
		data.append([float(x) for x in p[1:]])

	return rownames, colnames, data

def pearson(v1, v2):
	sum1 = sum(v1)
	sum2 = sum(v2)

	sum1sq = sum([pow(v, 2) for v in v1])
	sum2sq = sum([pow(v, 2) for v in v2])

	pSum = sum([v1[i]*v2[i] for i in range(len(v1))])

	num = pSum - (sum1*sum2/len(v1))
	den = sqrt((sum1sq - pow(sum1, 2)/len(v1)) * (sum2sq - pow(sum2, 2)/len(v1)))

	if den == 0:
		return 0

	return 1.0 - num/den

#hierarchical clustering algorithm
def hcluster(rows, distance=pearson):
	distances = {}
	currClustId = -1

	# clusters are intially just the rows
	clust = [bicluster(rows[i], id=i) for i in range(len(rows))]
	#print(len(clust));exit(1)
	while len(clust) > 1:
		lowestpair = (0, 1)
		closest = distance(clust[0].vec, clust[1].vec)

		for i in range(len(clust)):
			for j in range(i+1, len(clust)):
				if (clust[i].id, clust[j].id) not in distances:
					distances[(clust[i].id, clust[j].id)] = distance(clust[i].vec, clust[j].vec)

				d = distances[(clust[i].id, clust[j].id)]

				if d < closest:
					closest = d
					lowestpair = (i, j)

		mergeVec = [(clust[lowestpair[0]].vec[i] + clust[lowestpair[1]].vec[i])/2.0 for i in range(len(clust[0].vec))]

		newCluster = bicluster(mergeVec, left=clust[lowestpair[0]], right=clust[lowestpair[1]], distance=closest, id=currClustId)

		currClustId = -1
		del clust[lowestpair[1]]
		del clust[lowestpair[0]]
		clust.append(newCluster)

	return clust[0]

def printCluster(clust, label=None, n=0):
	for i in range(n):
		print ' ',
	
	if clust.id < 0:
		print '-'
	else:
		if label == None:
			print clust.id
		else:
			print label[clust.id]

	if clust.left != None:
		printCluster(clust.left, label=label, n=n+1)
	if clust.right != None:
		printCluster(clust.right, label=label, n=n+1)

def getHeight(clust):
	if clust.left == None and clust.right == None:
		return 1

	return getHeight(clust.left) + getHeight(clust.right)

def getDepth(clust):
	if clust.left == None and clust.right == None:
		return 0

	return max(getDepth(clust.left), getDepth(clust.right)) + clust.distance

def drawDendrogram(clust, label, img_file='clusters.jpg'):
	h = getHeight(clust)*20
	w = 1200
	depth = getDepth(clust)

	scaling = float(w - 150)/depth

	img = Image.new('RGB', (w, h), (255, 255, 255))
	draw = ImageDraw.Draw(img)

	draw.line((0, h/2, 10, h/2), fill = (255, 0, 0))

	drawNode(draw, clust, 10, (h/2), scaling, label)
	img.save(img_file, 'JPEG')

def drawNode(draw, clust, x, y, scaling, label):
	if clust.id < 0:
		h1 = getHeight(clust.left)*20
		h2 = getHeight(clust.right)*20
		top = y - (h1 + h2)/2
		bottom = y + (h1 + h2)/2

		# Line length
		ll = clust.distance*scaling

		# Vertical line from this cluster to children
		draw.line((x, top + h1/2, x, bottom - h2/2), fill=(255, 0, 0))

		# Horizontal line to left item
		draw.line((x, top+h1/2, x+ll, top+h1/2), fill=(255, 0, 0))

		# Horizontal line to right item
		draw.line((x, bottom-h2/2, x+ll, bottom-h2/2), fill=(255, 0, 0))

		# Call the function to draw the left and right nodes
		drawNode(draw, clust.left, x+ll, top+h1/2, scaling, label)
		drawNode(draw, clust.right, x+ll, bottom-h2/2, scaling, label)
	else:
		# If this is an endpoint, draw the item label
		draw.text((x+5, y-7), label[clust.id], (0,0,0))

def rotateMatrix(data):
	newData = []
	for i in range(len(data[0])):
		newrow = [data[j][i] for j in range(len(data))]
		newData.append(newrow)

	return newData

def kMeansCluster(rows, distance=pearson, k=4):
	# Determine the minimum and maximum values for each point
	ranges = [(min([row[i] for row in rows]), max([row[i] for row in rows])) for i in range(len(rows[0]))]
	
	# Create k randomly placed centroids
	clusters = [[random.random()*(ranges[i][1] - ranges[i][0]) + ranges[i][0] for i in range(len(rows[0]))] for j in range(k)]
	'''
	for cluster in clusters:
		#print(cluster)
		print([round(clst, 2) for clst in cluster])
		print('-'*300)
	exit(1)
	'''
	lastMatches = None
	for t in range(100):
		print 'Iteration %d' %t
		bestMatches = [[] for i in range(k)]

		# Find which centroid is the closest for each row
		for j in range(len(rows)):
			row = rows[j]
			bestMatch = 0

			for i in range(k):
				d = distance(clusters[i], row)

				if i != bestMatch and d < distance(clusters[bestMatch], row):
					bestMatch = i

			bestMatches[bestMatch].append(j)

		#print(bestMatches);exit(1)
		# If the results are the same as last time, this is complete
		if bestMatches == lastMatches:
			break

		lastMatches = bestMatches

		# Move the centroids to the average of their members
		for i in range(k):
			avgs = [0.0]*len(rows[0])

			if len(bestMatches[i]) > 0:

				for rowId in bestMatches[i]:
					for m in range(len(rows[rowId])):
						avgs[m] += rows[rowId][m]
				
				for j in range(len(avgs)):
					avgs[j] /= len(bestMatches[i])

				#update clusters with the new centroids(averages)
				clusters[i] = avgs
	
	return bestMatches


(blognames, words, data) = readfile('/media/sumit/Storage/Downloads/Machine-Learning-Concepts/2-Discovering-Groups/blogdata.txt')

#hierarchical clustering algorithm
clust = hcluster(data)

rdata = rotateMatrix(data)
rclust = hcluster(rdata)

printCluster(clust, label=blognames)
drawDendrogram(clust, blognames, img_file='blogclust.jpg')
drawDendrogram(rclust, words, img_file='wordclust.jpg')

# K Means clustering Algorithm
kclust = kMeansCluster(data, k=4)

for clust in kclust:
	print([blognames[r] for r in clust])
	print('='*300)