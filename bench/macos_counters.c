// Read the encoder's final process counters while it is an unreaped child.
// No root privileges, Instruments, or private PMU configuration is needed.
#include <errno.h>
#include <libproc.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

static double now(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &t)) {
        perror("clock_gettime");
        exit(2);
    }
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s /path/to/weft [arguments...]\n", argv[0]);
        return 2;
    }
    pid_t pid;
    double start = now();
    int error = posix_spawn(&pid, argv[1], NULL, NULL, argv + 1, environ);
    if (error) {
        fprintf(stderr, "posix_spawn: %s\n", strerror(error));
        return 2;
    }
    siginfo_t info = {0};
    while (waitid(P_PID, pid, &info, WEXITED | WNOWAIT)) {
        if (errno != EINTR) {
            perror("waitid");
            return 2;
        }
    }
    double wall = now() - start;
    struct rusage_info_v6 counters = {0};
    int result = proc_pid_rusage(pid, RUSAGE_INFO_V6, (rusage_info_t *)&counters);
    if (result) perror("proc_pid_rusage");
    int status;
    struct rusage usage;
    while (wait4(pid, &status, 0, &usage) < 0) {
        if (errno != EINTR) {
            perror("wait4");
            return 2;
        }
    }
    if (result || !WIFEXITED(status) || WEXITSTATUS(status)) return 1;
    if (!counters.ri_instructions || !counters.ri_cycles) {
        fprintf(stderr, "Hardware instruction/cycle counters are unavailable\n");
        return 1;
    }
    double cpu = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec * 1e-6
               + usage.ru_stime.tv_sec + usage.ru_stime.tv_usec * 1e-6;
    // macOS reports ru_maxrss in bytes. P-core counts are a subset of
    // process counts, not counters for an additional execution interval.
    fprintf(stderr,
            "COUNTERS {\"wall\":%.9f,\"cpu\":%.9f,\"instructions\":%llu,"
            "\"cycles\":%llu,\"p_instructions\":%llu,\"p_cycles\":%llu,\"maxrss\":%ld}\n",
            wall, cpu, (unsigned long long)counters.ri_instructions,
            (unsigned long long)counters.ri_cycles,
            (unsigned long long)counters.ri_pinstructions,
            (unsigned long long)counters.ri_pcycles, usage.ru_maxrss);
    return 0;
}
