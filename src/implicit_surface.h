#ifndef __IMPLICIT_SURFACE_H__
#define __IMPLICIT_SURFACE_H__

#include "mesh.h"
#include "vector.h"
#include "Expression.h"
#include "Evaluate.h"
#include <vector>
#include <string>

class ImplicitSurface : public Mesh {
	int functionsCount;
	Expression ** expressions;
	Vector gridStart; // Position the grid in 'the function space' 
	Vector gridSize;  // Number of Cells in x,y,z dims
	double cellSize;  // Using Cube Cells with side - cellSize

	void generateMesh();

	void fillProperties(ParsedBlock& pb)
	{
		pb.getIntProp("functionsCount", &functionsCount);
		expressions = new Expression*[functionsCount];
		for (int i = 0; i < functionsCount; i++){
			char function[256];
			pb.getStringProp("function", &function[0]);
			expressions[i] = new Expression(function);
		}
		pb.getVectorProp("gridSize", &gridSize);
		pb.getDoubleProp("cellSize", &cellSize);
		pb.getVectorProp("gridStart", &gridStart);

		// Inherited from Mesh
		pb.getBoolProp("faceted", &faceted);
		pb.getBoolProp("backfaceCulling", &backfaceCulling);
		pb.getBoolProp("useKDTree", &useKDTree);
		pb.getBoolProp("useSAH", &useSAH);
		pb.getBoolProp("autoSmooth", &autoSmooth);

		// Marching Cubes
		generateMesh();
	}
};

#endif