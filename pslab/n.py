
def more_than_or_eq_two_nines(n):
  with_nines = str(n)
  without_nines = str(n).replace("9", "")
  return (len(with_nines) - len(without_nines)) >= 2

n = 0

for i in range(0, 1000000, 9):
  if more_than_or_eq_two_nines(i):
    n += 1

print(n)

print(sum([((len(str(i)) - len(str(i).replace("9", ""))) >= 2) for i in range(0, 1000000, 9)]))
