def Formatter(X_in):
  X = X_in.copy()
  X["Title"] = X["Name"].str.extract(r',\s*([^\.]+)\.')
  allowed_titles = ["Mr", "Mrs", "Miss", "Master"]
  X["Title"] = X["Title"].apply(
      lambda x: x if x in allowed_titles else "Rare"
  )
  X.drop(columns=['Ticket','Cabin','PassengerId','Name'], inplace=True)
  return X