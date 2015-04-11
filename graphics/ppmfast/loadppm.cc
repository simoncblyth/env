// clang++  loadppm.cc -o /tmp/loadppm


// http://josiahmanson.com/prose/optimize_ppm/

#include "stdio.h"
#include "stdlib.h"
#include <vector>

struct RGB
{
    unsigned char r, g, b;
};

struct ImageRGB
{
    int width, height;
    std::vector<RGB> data;
};

int loadfile(std::vector<unsigned char>& buf, const char* path)
{
    FILE *fp = fopen(path, "rb");
    if (!fp)
    {
        printf("load_file failed to open %s \n", path);
        return 1;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    buf.resize(size+1);
    fread((char*)&buf[0], 1, size, fp);
    buf[size] = ' ';

    fclose(fp);
    return 0;
}


void eat_white(unsigned char*& ptr, const unsigned char* end)
{
    for (; ptr != end; ++ptr)
    {
        if (*ptr != '\n' && *ptr != '\r' && *ptr != '\t' && *ptr != ' ')
            break;
    }
}

void eat_line(unsigned char*& ptr, const unsigned char* end)
{
    for (; ptr != end; ++ptr)
    {
        if (*ptr == '\n')
            break;
    }
    ptr++;
}

void eat_comment(unsigned char* &ptr, const unsigned char* end)
{
    while (ptr != end)
    {
        eat_white(ptr, end);
        if (*ptr != '#')
            break;
        eat_line(ptr, end);
    }
}

void eat_token(unsigned char* &ptr, const unsigned char* end)
{
    for (; ptr != end; ++ptr)
    {
        if (*ptr == '\n' || *ptr == '\r' || *ptr == '\t' || *ptr == ' ')
            break;
    }
}

int get_int(unsigned char*&ptr, const unsigned char* end)
{
    eat_white(ptr, end);
    int v = atoi((char*)ptr);
    eat_token(ptr, end);
    return v;
}

void loadppm(ImageRGB &img, const char* path)
{
    std::vector<unsigned char> buf;
    if (loadfile(buf, path)) return;
    
    unsigned char* ptr = &buf[0];
    const unsigned char* end = ptr + buf.size();

    // get type of file
    eat_comment(ptr, end);
    eat_white(ptr, end);
    int mode = 0;
    if (ptr + 2 < end && ptr[0] == 'P')
    {
        mode = ptr[1] - '0';
        ptr += 2;
    }
    
    eat_comment(ptr, end);
    img.width = get_int(ptr, end);

    eat_comment(ptr, end);
    img.height = get_int(ptr, end);
    
    eat_comment(ptr, end);
    int bits = get_int(ptr, end);

    // load image data
    img.data.resize(img.width * img.height);

    if (mode == 6)
    {
        ptr++;
        memcpy(&img.data[0], ptr, img.data.size() * 3);
    }
    else if (mode == 3)
    {
        for (int i = 0; i < img.data.size(); i++)
        {
            img.data[i].r = get_int(ptr, end);
            img.data[i].g = get_int(ptr, end);
            img.data[i].b = get_int(ptr, end);
        }
    }
}




int main()
{
    const char* path = "/tmp/teapot.ppm";

    ImageRGB img ;
    loadppm(img, path );

    printf("loaded %s into img.data of size %lu width %d height %d \n", path, img.data.size(), img.width, img.height );

    return 0 ;
}

