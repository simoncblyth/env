/**
curl_check_1.cc
=================

**/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <curl/curl.h>

struct UploadData {
    const char *data;
    size_t size;
};

struct WriteData {
    char *buffer;
    size_t size;
};

size_t readfunction_callback(void *buffer, size_t size, size_t nitems, void *userdata) 
{
    struct UploadData* upload_data = (struct UploadData *)userdata;
    size_t copy_size = size * nitems;

    // Determine how many bytes to copy
    if (copy_size > upload_data->size) {
        copy_size = upload_data->size;
    }

    // Copy the data into libcurl's buffer
    memcpy(buffer, upload_data->data, copy_size);

    upload_data->data += copy_size;  // move data pointer
    upload_data->size -= copy_size;  // decrease remaining size

    return copy_size;
}



size_t writefunction_callback(char *ptr, size_t size, size_t nmemb, void *userdata) 
{
    struct WriteData* write_data = (struct WriteData *)userdata;
    size_t new_len = write_data->size + (size * nmemb);
    char *new_buffer = (char*)realloc(write_data->buffer, new_len + 1);

    if (new_buffer == NULL) {
        // Realloc failed, a real-world app would handle this more robustly
        fprintf(stderr, "realloc() failed!\n");
        return 0; // Abort transfer
    }

    // Update the buffer pointer and size
    write_data->buffer = new_buffer;
    memcpy(&(write_data->buffer[write_data->size]), ptr, size * nmemb);
    write_data->buffer[new_len] = '\0';
    write_data->size = new_len;

    return size * nmemb;
}

int main(void) 
{
    CURL *curl;
    CURLcode res;

    // Initialize the upload data
    const char *upload_payload = "Hello, server!";
    struct UploadData upload_data = {
        .data = upload_payload,
        .size = strlen(upload_payload)
    };

    // Initialize the write data
    struct WriteData write_data = {
        .buffer = (char*)malloc(1), // Start with a 1-byte buffer
        .size = 0
    };
    write_data.buffer[0] = '\0';

    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();


    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, "https://httpbin.org/post");
        curl_easy_setopt(curl, CURLOPT_POST, 1L);

        curl_easy_setopt(curl, CURLOPT_READFUNCTION, readfunction_callback);
        curl_easy_setopt(curl, CURLOPT_READDATA, &upload_data);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)upload_data.size);

        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writefunction_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &write_data);

        // Perform the request
        res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        } else {
            printf("Received response: \n%s\n", write_data.buffer);
        }

        // Clean up
        curl_easy_cleanup(curl);
        free(write_data.buffer);
    }

    curl_global_cleanup();
    return 0;
}





