# Set output
set terminal pdfcairo size 17in,10in enhanced font "Verdana,35"
set output 'fig-2.pdf'

# Formatting
# set title "BFS RST depth vs GConn RST depth Comparison" font ",30"
set ylabel "RST Depth" font "Verdana,45"
set xlabel "Dataset" font "Verdana,45"
set grid y
set key top left

# REMOVE TOP AND RIGHT TICS
# nomirror ensures tics don't show up on the opposite axes
set xtics nomirror rotate by 45 right font "Verdana,45"
set ytics nomirror font "Verdana,45"

# Remove the small scale tics specifically (minor tics)
unset mxtics
unset mytics

# Histogram Style
set style data histograms
set style histogram cluster gap 1
set style fill solid 0.8 noborder
set boxwidth 0.9

# Log Scale is necessary for values ranging from 6 to 550,000
set logscale y

set xrange [0:*]
# Define the data using a heredoc (requires Gnuplot 5.0+)
$MyData << EOD
Dataset  BFS_Depth  GConn_Depth
WB       973        971
AS       757        756
HT       157        156
CD       14         60
SO       23581      23579
RU       6143       40790
LJ       1877       1876
K20      253378     253379
EU       19932      59347
K21      553161     553158
CO       6          32
UK       38360      38358
EOD

# Plotting
# Column 2 = BFS, Column 3 = GConn, xtic(1) = Dataset labels from Col 1
plot $MyData using 2:xtic(1) title "BFS Depth" linecolor rgb "#1f77b4", \
     $MyData using 3 title "GConn Depth" linecolor rgb "#d62728"