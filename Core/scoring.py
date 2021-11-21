def calculate_scores(neaarest_neighbs_words,neaarest_neighbs_labels):
  score_dict = {
      'Religion/creed': 0,
      'Race/ethnicity': 0,
      'Gender': 0,
      'Sexual Orientation': 0,
      'Physical/disability': 0,
  }

  for i in range(0,len(neaarest_neighbs_words)):
    score = 50-i
    # print(score,neaarest_neighbs_words[i],neaarest_neighbs_labels[i])
    score_dict[neaarest_neighbs_labels[i]]=score_dict[neaarest_neighbs_labels[i]]+score

  score_max = (len(neaarest_neighbs_words)*(len(neaarest_neighbs_words)-1))/2
  normalized_score_dict = score_dict.copy()
  for k in score_dict.keys():
    if score_dict[k] ==0:
      del normalized_score_dict[k]
    else:
      normalized_score_dict[k] = round((score_dict[k]/score_max),3)

  # print(score_dict)
  print(normalized_score_dict)

  return normalized_score_dict