/** 
 * \file npyreader.c
 * \brief functions to read a .npy numpy file.
 *
 * \author: Jordi Castells Sala
 */

#include <stdlib.h>
#include <string.h>
#include "npyreader.h"


int NPYERRNO = 0;

long fsize(FILE *fp){
	fseek(fp, 0, SEEK_END);
	long size = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	return size;
}

int32_t* retrieve_npy_int32(FILE* fp){
	uint16_t hsize = retrieve_HEADER_LEN(fp);
	int skstart = MAGIC_SIZE + 4 + hsize;

	long size = fsize(fp);

	int err = fseek(fp, skstart, SEEK_SET);

	int32_t* array_i32t = malloc(size - skstart);
	if (array_i32t == NULL){
		NPYERRNO = NPYERR_NOMEM;
		return NULL;
	}

	int nread = fread(array_i32t, sizeof(int32_t), (size - skstart)/sizeof(int32_t), fp);
	if(!nread) NPYERRNO = NPYERR_FREAD;

	//printf(">%lu\n",(size-skstart)/sizeof(float32_t));

	return array_i32t;
}


float32_t* retrieve_npy_float32(FILE* fp){
	uint16_t hsize = retrieve_HEADER_LEN(fp);
	int skstart = MAGIC_SIZE + 4 + hsize;

	long size = fsize(fp);

	int err = fseek(fp, skstart, SEEK_SET);

	float32_t* array_f32t = malloc(size - skstart);
	if (array_f32t == NULL){
		NPYERRNO = NPYERR_NOMEM;
		return NULL;
	}

	int nread = fread(array_f32t, sizeof(float32_t), (size - skstart)/sizeof(float32_t), fp);
	if(!nread) NPYERRNO = NPYERR_FREAD;

	//printf(">%lu\n",(size-skstart)/sizeof(float32_t));

	return array_f32t;
}

float64_t* retrieve_npy_float64(FILE* fp){
	uint16_t hsize = retrieve_HEADER_LEN(fp);
	int skstart = MAGIC_SIZE + 4 + hsize;

	long size = fsize(fp);

	int err = fseek(fp, skstart, SEEK_SET);

	float64_t* array_f64t = malloc(size - skstart);
	if (array_f64t == NULL){
		NPYERRNO = NPYERR_NOMEM;
		return NULL;
	}

	int nread = fread(array_f64t, sizeof(char), size - skstart, fp);
	if(!nread) NPYERRNO = NPYERR_FREAD;

	return array_f64t;
}


uint32_t* get_shape_file(FILE* fp){
	char* dict =  retrieve_python_dict(fp);
	uint32_t* sizes = get_shape(dict);

	free(dict);
	return sizes;
}


/*
   dict {'descr': '<f4', 'fortran_order': False, 'shape': (38, 2), } 
*/

char* get_descr(char* dictstring)
{
	char* colon  = strchr(strstr(dictstring, "'descr'"), ':');
    char* dtype = NULL ; 
    char* begin = NULL ; 
    char* end = NULL ; 

    int nquote = 0 ;
    char* ptr = colon ; 
    while(*ptr){
       if(*ptr == '\''){
           nquote++ ; 
           if(nquote == 1) begin=ptr + 1 ;
           if(nquote == 2) end = ptr ; 
           if( nquote >= 2) break ; 
       }  
       ptr++ ; 
    }

    if( begin != NULL && end != NULL){
         size_t len = end - begin ; 
         dtype = (char*)malloc(sizeof(char) * 16); 
         strncpy(dtype, begin, len); 
         dtype[len] = '\0';
    }
    return dtype ;
}


char* get_shapestr(char* dictstring)
{
	char* colon  = strchr(strstr(dictstring, "'shape'"), ':');
    char* shape = NULL ; 
    char* begin = NULL ; 
    char* end = NULL ; 

    int nquote = 0 ;
    char* ptr = colon ; 
    while(*ptr){
       if(*ptr == '(') begin = ptr + 1 ;
       if(*ptr == ')') end = ptr ;
       if( end != NULL ) break ; 
       ptr++ ; 
    }

    if( begin != NULL && end != NULL){
         size_t len = end - begin ; 
         shape = (char*)malloc(sizeof(char) * 16); 
         strncpy(shape, begin, len); 
         shape[len] = '\0';
    }
    return shape ;
}





uint32_t* get_shape(char* dictstring){
	char* sizstart = strchr(strstr(dictstring, "'shape'"),'(');
	sizstart++; //remove the (

	//Anything bigger than 5 dimensions will not be accepted
	uint32_t* sizes = malloc(sizeof(uint32_t) * 5);
	if(sizes == NULL){
		NPYERRNO = NPYERR_NOMEM;
		return NULL;
	}


	char *numbers = "0123456789 ";
	size_t len;
	int i = 0;

	//Allocate a temporal string to hold results
	char* sizestr = calloc(100, sizeof(char));

	while((len = strspn(sizstart, numbers))!= 0){
		strncpy(sizestr, sizstart, len);
		sizestr[len] = '\0';
		sizes[i] = (uint32_t)atoi(sizestr);
		sizstart += len+1;
		i++;
	}
	sizes[i] = -1; //The array finishes with -1

	free(sizestr);

	return sizes;
}

char* retrieve_python_dict(FILE* fp){
	uint16_t hsize = retrieve_HEADER_LEN(fp);

	char* dict = malloc(sizeof(char) * (hsize + 1));
	if(dict == NULL){
		NPYERRNO = NPYERR_NOMEM;
		return NULL;
	}

	int err = fseek(fp, DICT_SEEK, SEEK_SET);
	if(err == -1) NPYERRNO = NPYERR_FSEEK;
	int nread = fread(dict, sizeof(char), hsize, fp);
	if(!nread) NPYERRNO = NPYERR_FREAD;

	dict[hsize] = '\0';

	return dict;
}

char* retrieve_magic(FILE* fp){
	char* magic = malloc(sizeof(char) * MAGIC_SIZE);
	if(magic == NULL){
		NPYERRNO = NPYERR_NOMEM;
		return NULL;
	}

	rewind(fp);
	int nread = fread(magic, sizeof(char), MAGIC_SIZE, fp);
	if(!nread) NPYERRNO = NPYERR_FREAD;
	

	return magic;
}

uint8_t retrieve_version_number_major(FILE* fp){
	uint8_t val;

	int err = fseek(fp, VERSION_SEEK, SEEK_SET);
	if (err)
		return -1;

	int nread = fread(&val, sizeof(uint8_t), 1, fp);
	if(!nread) NPYERRNO = NPYERR_FREAD;

	return val;
}

uint8_t retrieve_version_number_minor(FILE* fp){
	uint8_t val;

	int err = fseek(fp, MINOR_VERSION_SEEK, SEEK_SET);
	if (err)
		return -1;

	int nread = fread(&val, sizeof(uint8_t), 1, fp);
	if(!nread) NPYERRNO = NPYERR_FREAD;

	return val;
}

uint16_t retrieve_HEADER_LEN(FILE *fp){
	uint16_t val;

	int err = fseek(fp, HEADER_LEN_SEEK, SEEK_SET);
	if (err)
		return -1;

	int nread = fread(&val, sizeof(uint16_t), 1, fp);
	if(!nread) NPYERRNO = NPYERR_FREAD;

	return val;
}

float32_t** allocate_matrix_f32(float32_t* data, int rows, int cols){
	float32_t** matrix = malloc(rows * sizeof(float32_t*));
	if(matrix == NULL){
		NPYERRNO = NPYERR_NOMEM;
		return NULL;
	}

	int i;
	for(i=0; i<rows; i++){
    	matrix[i]= (float32_t *) malloc(cols*sizeof(float32_t));
		if(matrix[i] == NULL){
			NPYERRNO = NPYERR_SUBNOMEM;
			return NULL;
		}
	}

	int r,c;
	for(r=0; r<rows; r++){
		for(c=0; c<cols; c++){
			//printf("[%d,%d,%d]>%f\n",r,c,r*cols+c,data[r*cols+c]);
			matrix[r][c] = data[r*cols + c];
		}
	}

	return matrix;
}

int get_number_of_sizes(uint32_t* sizes){
	int nsizes = 0;
	int i = 0;
	for(i=0; sizes[i] != -1; i++)
		nsizes++;

	return nsizes;

}


float32_t** get_2D_matrix_f32(FILE* fp){
	char* dict = retrieve_python_dict(fp);
	if (dict == NULL) return NULL;

	uint32_t* sizes = get_shape(dict);
	free(dict);
	if (sizes == NULL) {
		NPYERRNO = NPYERR_GETSHAPE;
		return NULL;
	}

	int nsizes = get_number_of_sizes(sizes);
	if (nsizes != 2) {
		NPYERRNO = NPYERR_BADSIZENUMBER;
		return NULL;
	}

	float32_t*  data = retrieve_npy_float32(fp);
	if (data == NULL) {
		NPYERRNO = NPYERR_READDATA;
		return NULL;
	}

	float32_t** matrix = allocate_matrix_f32(data, sizes[0], sizes[1]);
	if (matrix == NULL){
		//NPYERROR is already set by the allocate_matrix function
		return NULL;
	}

	free(data);
	free(sizes);

	return matrix;
}

float32_t* get_1D_array_f32(FILE* fp){
	char* dict = retrieve_python_dict(fp);
	if (dict == NULL) {
		NPYERRNO = NPYERR_READDICT;
		return NULL;
	}

	uint32_t* sizes = get_shape(dict);
	free(dict);

	if (sizes == NULL) {
		NPYERRNO = NPYERR_GETSHAPE;
		return NULL;
	}

	int nsizes = get_number_of_sizes(sizes);
	free(sizes);

	if (nsizes != 1){
		NPYERRNO = NPYERR_BADSIZENUMBER;
		return NULL;
	}

	float32_t*  data = retrieve_npy_float32(fp);
	if(data == NULL){
		NPYERRNO = NPYERR_READDATA;
		return NULL;
	}

	return data;
}

int npyerrno(){
	int err = NPYERRNO;
	NPYERRNO = 0;
	return err;
}
