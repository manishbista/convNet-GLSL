#include "../header/mesh.h"


mesh::mesh(std::vector<vertexData>* vd,std::vector<unsigned int>* id,std::vector<textureData>* td)
{
	data=*vd;
	indices=*id;
	if(td)
		textures=*td;
	
	glGenBuffers(1,&VBO);
	glBindBuffer(GL_ARRAY_BUFFER,VBO);
	glBufferData(GL_ARRAY_BUFFER,data.size()*sizeof(vertexData),
	&data[0],GL_STATIC_DRAW);
	
	
	glGenBuffers(1,&IND);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,IND);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER,indices.size()*sizeof(unsigned int),&indices[0],GL_STATIC_DRAW);
	
	glBindBuffer(GL_ARRAY_BUFFER,0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0);
	
}

mesh::~mesh()
{
	glDeleteBuffers(1,&VBO);
	glDeleteBuffers(1,&IND);
}

void mesh::draw(unsigned int programId)
{
	//attribute vec3 vertex
	int vertex=glGetAttribLocation(programId,"vertex"); //0
	int normal=glGetAttribLocation(programId,"normal"); //1
	int tangent=glGetAttribLocation(programId,"tangent"); //2
	int color=glGetAttribLocation(programId,"color"); //3
	int UV=glGetAttribLocation(programId,"UV"); //4
	
	//texture0
	//texture1...
	std::string str="texture";
	//std::cout<<"tsize"<<textures.size()<<std::endl;
	for(int i=0;i<textures.size();i++)
	{
		int j = i + 1;
		glActiveTexture(GL_TEXTURE0+j);
		glBindTexture(GL_TEXTURE_2D,textures[i].id);
		glUniform1i(glGetUniformLocation(programId,(str+(char)(i+'0')).c_str()),j);
	}
	
	glBindBuffer(GL_ARRAY_BUFFER,VBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,IND);
	
	glEnableVertexAttribArray(vertex);
	glVertexAttribPointer(vertex,3,GL_FLOAT,GL_FALSE,sizeof(vertexData),0);
	
	glEnableVertexAttribArray(normal);
	glVertexAttribPointer(normal,3,GL_FLOAT,GL_FALSE,sizeof(vertexData),(void*)(3*sizeof(float)));
	
	glEnableVertexAttribArray(tangent);
	glVertexAttribPointer(tangent,3,GL_FLOAT,GL_FALSE,sizeof(vertexData),(void*)(6*sizeof(float)));
	
	glEnableVertexAttribArray(color);
	glVertexAttribPointer(color,3,GL_FLOAT,GL_FALSE,sizeof(vertexData),(void*)(9*sizeof(float)));
	
	glEnableVertexAttribArray(UV);
	glVertexAttribPointer(UV,2,GL_FLOAT,GL_FALSE,sizeof(vertexData),(void*)(12*sizeof(float)));
	
	glDrawElements(GL_TRIANGLES,indices.size(),GL_UNSIGNED_INT,0);
	
	glDisableVertexAttribArray(vertex);
	glDisableVertexAttribArray(normal);
	glDisableVertexAttribArray(tangent);
	glDisableVertexAttribArray(color);
	glDisableVertexAttribArray(UV);
	glBindBuffer(GL_ARRAY_BUFFER,0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0);	
}
