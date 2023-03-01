# Parallel-blurhash
This is a parallel version of the popular blurhash algotrithm made using CUDA C/C++

## What is BlurHash ?
In short, BlurHash takes an image, and gives you a short string (only 20-30 characters!) that represents the placeholder for this image. You do this on the backend of your service, and store the string along with the image. When you send data to your client, you send both the URL to the image, and the BlurHash string. Your client then takes the string, and decodes it into an image that it shows while the real image is loading over the network. The string is short enough that it comfortably fits into whatever data format you use. For instance, it can easily be added as a field in a JSON object. 

## Perfomance
The CPU version takes about 13 seconds for an image of size 725x900 with xComponents and yComponents as 8 and 8 where as the GPU version takes around 3.5 seconds for the same.

## Source for CPU implementation
https://github.com/woltapp/blurhash
