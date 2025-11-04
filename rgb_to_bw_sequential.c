//add the packed
//difine the size   
//add int x and y for pisstion in the pixal screen
//use maloc for each R,G,B and remeber to free them 
//add a error for null for maoc if error in maloc ad a IF and error masge 
//intialaize the arry with radnom x,y value uinsg a loop
// and intialize a float for SUM 
// add the logic code for BW +++++++++++++++++++++++++
// now divide the sum/pixal x to y

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WIDTH 800
#define HEIGHT 600

int main(void) {
    int x, y;

    // Seed the random number generator
    srand(1234);  // or srand(time(NULL)) for true randomness

    // Allocate 2D arrays dynamically to avoid stack overflow
    float (*Red)[HEIGHT]   = malloc(sizeof(float[WIDTH][HEIGHT]));
    float (*Green)[HEIGHT] = malloc(sizeof(float[WIDTH][HEIGHT]));
    float (*Blue)[HEIGHT]  = malloc(sizeof(float[WIDTH][HEIGHT]));
    float (*BW)[HEIGHT]    = malloc(sizeof(float[WIDTH][HEIGHT]));

    if (!Red || !Green || !Blue || !BW) {
        perror("Memory allocation failed");
        return 1;
    }

    // Initialize the RGB arrays with random intensity values [0, 255]
    for (x = 0; x < WIDTH; ++x) {
        for (y = 0; y < HEIGHT; ++y) {
            Red[x][y]   = rand() % 256;
            Green[x][y] = rand() % 256;
            Blue[x][y]  = rand() % 256;
        }
    }

    double sum = 0.0;

    // --- Sequential conversion to black and white ---
    for (x = 0; x < WIDTH; ++x) {
        for (y = 0; y < HEIGHT; ++y) {
            BW[x][y] = 0.21f * Red[x][y] + 0.72f * Green[x][y] + 0.07f * Blue[x][y];
            sum += BW[x][y];
        }
    }

    double average = sum / (WIDTH * HEIGHT);

    printf("Conversion completed (Sequential version)\n");
    printf("Average BW pixel value = %.2f\n", average);

    // Optional: print a few sample pixel results
    printf("Sample pixels:\n");
    for (int i = 0; i < 3; ++i)
        printf("BW[%d][%d] = %.2f\n", i, i, BW[i][i]);

    free(Red);
    free(Green);
    free(Blue);
    free(BW);

    return 0;
}

