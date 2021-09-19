input = getDirectory("select")
list = getFileList(input);
for (i = 0; i < list.length; i++){
	file=File.openAsString( input + list[i]);

	lines = split(file, "\n");
	n_lines = lengthOf(lines);
	xcoor=newArray(n_lines);
	ycoor=newArray(n_lines);
	for (lineNum=0;lineNum<lines.length;lineNum++){
	//-- Debug Only
	//print(lines[lineNum]);
	//-- extract the coordinates by splitting on tab
	pointCoord=split(lines[lineNum],",");
	//-- Draw the point
		
    xcoor[lineNum] = pointCoord[0];
    ycoor[lineNum] = pointCoord[1];
	}
	makeSelection("polygon", xcoor, ycoor);
	roiManager("Add");
	count = roiManager("count");
	roiManager("select", count-1);
	roiManager("Rename", "Cell"+i+1+"");

}
