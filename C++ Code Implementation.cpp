#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include<bits/stdc++.h>

using namespace std;

struct Point {
    double x, y, z;
};

struct Triangle {
    Point p1, p2, p3;
    Point edges[3];
};

double determinant(Point a, Point b, Point c) {
    return a.x * (b.y * c.z - b.z * c.y) -
           a.y * (b.x * c.z - b.z * c.x) +
           a.z * (b.x * c.y - b.y * c.x);
}

bool isLegalIntersection(double gamma, double delta) {
    return gamma >= 0 && gamma <= 1 && delta >= 0 && delta <= 1;
}

bool intersectionTest(Triangle A, Triangle B) {
    // Stage 1: Calculate beta parameters
    Point beta;
    for (int i = 0; i < 3; i++) {
        Point qi = A.edges[i];
        // if(i==0){
        //     cout << "qi: " << qi.x << " " << qi.y << " " << qi.z << endl;
        // }
        Point A_qi = { A.p1.x, A.p2.x, qi.x };
        double det_A_qi = determinant({A_qi.x, A_qi.y, A_qi.z}, {A_qi.y, A_qi.z, A_qi.x}, {A_qi.z, A_qi.x, A_qi.y});
        
        Point ri = { qi.x - A.p1.x, qi.y - A.p1.y, qi.z - A.p1.z };
        Point A_ri = { A.p1.x, A.p2.x, ri.x };
        double det_A_ri = determinant({A_ri.x, A_ri.y, A_ri.z}, {A_ri.y, A_ri.z, A_ri.x}, {A_ri.z, A_ri.x, A_ri.y});
        
        beta.x = -det_A_qi / det_A_ri;
        
        // Stage 2: Check for legal beta values
        if (beta.x < 0 || beta.x > 1) {
            return false;
        }
        
        // Stage 3: Construct intersection segment
        Point T = { A.p1.x + beta.x * qi.x, A.p1.y + beta.x * qi.y, A.p1.z + beta.x * qi.z };
        Point t = { A.edges[1].x - A.edges[0].x, A.edges[1].y - A.edges[0].y, A.edges[1].z - A.edges[0].z };
        
        // Stage 4: Check if intersection segment intersects triangle B or fully contained in B
        Point P, gamma, delta;
        for (int j = 0; j < 3; j++) {
            P = B.p1;
            Point p1 = B.edges[j];
            Point p2 = B.edges[(j + 1) % 3];
            Point p3 = { p2.x - p1.x, p2.y - p1.y, p2.z - p1.z }; // edge p3

            double det_p1 = determinant(P, p1, t);
            double det_p2 = determinant(T, t, p1);
            double det_p3 = determinant(P, p1, p3);
            double det_p4 = determinant(T, t, p3);

            double det_total = determinant(p3, {-t.x, -t.y, -t.z}, {det_p1, det_p2, det_p3});
            gamma.x = determinant(Point{det_p2, det_p1, det_p4}, Point{det_p3, det_p4, det_total}, Point{0,0,0}) / det_total;
            delta.x = determinant(Point{det_p2, det_p1, det_p3}, Point{det_p3, det_p4, det_total}, Point{0,0,0}) / det_total;

            if (isLegalIntersection(gamma.x, delta.x)) {
                return true;
            }
        }
    }
    
    return false;
}
std::vector<Triangle> readTrianglesFromOFF(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }

    std::string line;
    std::getline(file, line); // Read the "OFF" line

    int numVertices, numFaces, numEdges;
    file >> numVertices >> numFaces >> numEdges;

    std::vector<Point> vertices(numVertices);
    std::vector<Triangle> triangles;

    // Print the number of vertices, faces and edges
    std::cout << "Number of vertices: " << numVertices << std::endl;
    std::cout << "Number of faces: " << numFaces << std::endl;
    std::cout << "Number of edges: " << numEdges << std::endl;

    // Read vertices
    for (int i = 0; i < numVertices; i++) {
        file >> vertices[i].x >> vertices[i].y >> vertices[i].z;
    }

    // just printing the vertices top 5
    std::cout << std::fixed << std::setprecision(20);
    for (int i = 0; i < 5; i++) {
        cout << "Vertex: " << vertices[i].x << " " << vertices[i].y << " " << vertices[i].z << endl;
    }


    // Read faces
    for (int i = 0; i < numFaces; i++) {
        int numVerticesInFace;
        file >> numVerticesInFace;
        if (numVerticesInFace != 3) {
            std::cerr << "Non-triangle face encountered, skipping." << std::endl;
            file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        int index1, index2, index3;
        file >> index1 >> index2 >> index3;
        Triangle gg;
        gg.p1 = vertices[index1];
        gg.p2 = vertices[index2];
        gg.p3 = vertices[index3];

        // Triangle triangle = {vertices[index1], vertices[index2], vertices[index3]};

        // Calculate edges
        gg.edges[0] = {gg.p2.x - gg.p1.x, gg.p2.y - gg.p1.y, gg.p2.z - gg.p1.z};
        gg.edges[1] = {gg.p3.x - gg.p2.x, gg.p3.y - gg.p2.y, gg.p3.z - gg.p2.z};
        gg.edges[2] = {gg.p1.x - gg.p3.x, gg.p1.y - gg.p3.y, gg.p1.z - gg.p3.z};

        triangles.push_back(gg);
        // triangles.push_back(triangle);
    }

    file.close();
    return triangles;
}

int main() {
    auto start  = std::chrono::high_resolution_clock::now();
    std::vector<Triangle> trianglesA = readTrianglesFromOFF("./VH_F_renal_pyramid_L_a.off");
    std::vector<Triangle> trianglesB = readTrianglesFromOFF("./VH_F_renal_pyramid_L_b.off");
    // std::cout << "Triangles from file A:" << std::endl;
    // for (const auto& triangle : trianglesA) {
    //     std::cout << "Triangle: "
    //               << "(" << triangle.p1.x << ", " << triangle.p1.y << ", " << triangle.p1.z << "), "
    //               << "(" << triangle.p2.x << ", " << triangle.p2.y << ", " << triangle.p2.z << "), "
    //               << "(" << triangle.p3.x << ", " << triangle.p3.y << ", " << triangle.p3.z << ")"
    //               << std::endl;
    // }

    // std::cout << "Triangles from file B:" << std::endl;
    // for (const auto& triangle : trianglesB) {
    //     std::cout << "Triangle: "
    //               << "(" << triangle.p1.x << ", " << triangle.p1.y << ", " << triangle.p1.z << "), "
    //               << "(" << triangle.p2.x << ", " << triangle.p2.y << ", " << triangle.p2.z << "), "
    //               << "(" << triangle.p3.x << ", " << triangle.p3.y << ", " << triangle.p3.z << ")"
    //               << std::endl;
    // }
    // Check all combinations of triangles from A and B for intersection
    // cout << "Start time: " << start << endl;
    int A = 0;
    for (const auto& triangleA : trianglesA) {
        int B = 0;
        for (const auto& triangleB : trianglesB) {
            // cout<<triangleA.
            if(A==0 && B==0){
                cout << "Triangle A: " << triangleA.p1.x << " " << triangleA.p1.y << " " << triangleA.p1.z << endl;
                cout << "Triangle A: " << triangleA.p2.x << " " << triangleA.p2.y << " " << triangleA.p2.z << endl;
                cout << "Triangle A: " << triangleA.p3.x << " " << triangleA.p3.y << " " << triangleA.p3.z << endl;
                cout << "Triangle B: " << triangleB.p1.x << " " << triangleB.p1.y << " " << triangleB.p1.z << endl;
                cout << "Triangle B: " << triangleB.p2.x << " " << triangleB.p2.y << " " << triangleB.p2.z << endl;
                cout << "Triangle B: " << triangleB.p3.x << " " << triangleB.p3.y << " " << triangleB.p3.z << endl;
            }
            if (intersectionTest(triangleA, triangleB)) {
                std::cout << "Triangles intersect." << std::endl;
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                cout << "Time taken: " << elapsed.count() << endl;
                return 0; // Optional: stop at the first intersection
            }
            B++;
        }
        A++;
        if(A % 100 == 0){
            std::cout << "A: " << A << endl;
        }
    }
    std::cout << "A: " << A << endl;
    std::cout << "No intersections found." << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    cout << "Time taken: " << elapsed.count() << endl;
    return 0;
}