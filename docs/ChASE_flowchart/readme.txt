To convert the flowchart from pdf format to png format, use the
`convert` command after having made sure that ImageMagick software is
installed (use MacPorts). Then run the following command line

/opt/local/bin/convert -density 300 flow-chart-chase_standalone.pdf -quality 90 -colorspace RGB flow-chart-chase_standalone.png
