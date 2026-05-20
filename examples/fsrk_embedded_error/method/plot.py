import matplotlib.pyplot as plt


def get_csv_cols(fname):
    """
    Purpose: Convert csv to list of columns
    Preconditions: fname is a string pointing to a valid CSV file
    Returns: List of lists. Each item of main list is a column, each item of column is the entry as a string
    """
    f = open(fname, 'r')
    cols = []
    
    # Go through each row of the CSV
    for row in f:
        items = row.strip().split(',') # Grab entries from the row

        # Add to cols list``
        for n in range(len(items)):
            i = items[n]
            if len(cols) <= n: # Is there enough columns?
                cols.append([])
            cols[n].append(i)
    
    f.close()
    
    return cols

strings = get_csv_cols("results.csv")
t = [float(t) for t in strings[0]]
y = [float(y) for y in strings[1]]

# Plot it!
plt.figure()
plt.plot(t, y, label='y over time')
plt.legend()
plt.title("Y Position Over Time")

plt.savefig("graph.png")