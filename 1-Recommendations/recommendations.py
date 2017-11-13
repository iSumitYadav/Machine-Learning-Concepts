def loadMovieLens(path="/media/sumit/Storage/Downloads/Machine-Learning-Concepts/1-Recommendations/ml-latest-small"):
	movies = {}
	movies_csv = open(path+"/movies.csv")
	movies_csv.readline() #skip first line (field names)

	for line in movies_csv:
		(id, title) = line.split(',')[0:2]
		movies[id] = title
	#print(movies)

	prefs = {}
	ratings = open(path+"/ratings.csv")
	ratings.readline() #skip first line (field names)

	for line in ratings:
		(user, movieid, rating, ts) = line.split(',')
		'''if user == '2':
			print(prefs);exit(1)'''
		prefs.setdefault(user, {})
		prefs[user][movies[movieid]] = float(rating)
	return prefs

prefs = loadMovieLens()
print(prefs['87'])