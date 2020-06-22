import numpy as np

def getCoverageGraphData(occ_grid):
  all_lines = list(occ_grid.ax1.get_lines())
  line_data = []

  for line in all_lines:
    data = line.get_xydata()

    line_data.append(tuple(data.tolist()[0]))

  return line_data

def computeDoubleCoverage(occ_grid):
  # Double coverage is given as a percentage of covered cells
  grid = occ_grid.grid
  
  covered_cells_count = np.count_nonzero(grid > 0)
  double_covered_cells = np.count_nonzero(grid > 1)

  return double_covered_cells/covered_cells_count * 100

def computeCoverageScore(occ_grid):
  
  grid = occ_grid.grid
  
  covered_cells_count = np.count_nonzero(grid > 0)
  sum_of_coverages = np.sum(grid[np.where(grid > 0)])

  return sum_of_coverages/covered_cells_count