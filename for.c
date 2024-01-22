#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define Max(a,b) ((a)>(b)?(a):(b))
#define N 20000

// Создание глобальных переменных
int it, itmax = 100;
double maxeps = 0.1e-7;
double **A, **B;

// Объявление используемых функций
void init();
double relax();
void verify();

int main(int an, char **as) {
    // Выделение памяти для матриц
    A = (double **)malloc(N * sizeof(double *));
    B = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        A[i] = (double *)malloc(N * sizeof(double));
        B[i] = (double *)calloc(N, sizeof(double));
    }

    double start_time = omp_get_wtime(); // Время старта параллельной программы

    init(); // Заполнение матрицы 
    // Релаксация может завершиться раньше, если будет достигнута необходимая точность
    for (it = 1; it <= itmax; it++) if (relax() < maxeps) break;
    verify(); // Проверка результата

    // Вывод результата и времени работы
    printf("  Run time OpenMP: %fs\n  P = %d\n", omp_get_wtime() - start_time, omp_get_max_threads());

    // Освобождение памяти
    for (int i = 0; i < N; i++) {free(A[i]); free(B[i]);}
    free(A); free(B);

    return 0;
}

// Инициализация матрицы согласно условию задачи
void init() {
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i <= N - 1; i++) {
        for (int j = 0; j <= N - 1; j++) {
            if (i == 0 || i == N - 1 || j == 0 || j == N - 1) A[i][j] = 0.;
            else A[i][j] = (1. + i + j);
        }
    }
}

// Одна итерация метода релаксации
double relax() {
    double eps = 0.0; // Общая для всех потоков переменная
    // В зависимости от четности итерации обновляем значения элементов матрицы A или B
    if (it % 2) {
        #pragma omp parallel for reduction(max:eps) schedule(static) collapse(2)
        for (int i = 1; i <= N - 2; i++) {
            for (int j = 1; j <= N - 2; j++) {
                B[i][j] = (A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4.;
                eps = Max(eps, fabs(A[i][j] - B[i][j]));
            }
        }
    } else {
        #pragma omp parallel for reduction(max:eps) schedule(static) collapse(2)
        for (int i = 1; i <= N - 2; i++) {
            for (int j = 1; j <= N - 2; j++) {
                A[i][j] = (B[i - 1][j] + B[i + 1][j] + B[i][j - 1] + B[i][j + 1]) / 4.;
                eps = Max(eps, fabs(A[i][j] - B[i][j]));
            }
        }
    }
    
    return eps;
}

// Проверка результата
void verify() {
    double s = 0.0; // Общая для всех потоков переменная
    // В зависимости от четности отработанного количества итерации 
    // используем матрицу A или B для вычисления результата
    if (it % 2) {
        #pragma omp parallel for reduction(+:s) schedule(static) collapse(2)
        for (int i = 0; i <= N - 1; i++) {
            for (int j = 0; j <= N - 1; j++) {
                s = s + A[i][j] * (i + 1) * (j + 1) / (N * N);
            }
        }
    } else {
        #pragma omp parallel for reduction(+:s) schedule(static) collapse(2)
        for (int i = 0; i <= N - 1; i++) {
            for (int j = 0; j <= N - 1; j++) {
                s = s + B[i][j] * (i + 1) * (j + 1) / (N * N);
            }
        }
    }
    
    printf("  S = %f\n", s);
}