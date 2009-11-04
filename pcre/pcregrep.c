
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <pcre.h>

char enter_reverse_mode[] = "\33[7m";
char exit_reverse_mode[] = "\33[0m";

int main(int argc, char **argv)
{
  const char *pattern;
  const char *errstr;
  int erroffset;
  pcre *expr;
  char line[512];
  assert(argc == 2); /* XXX fixme */
  pattern = argv[1];
  if (!(expr = pcre_compile(pattern, 0, &errstr, &erroffset, 0))) {
    fprintf(stderr, "%s: %s\n", pattern, errstr);
    return EXIT_FAILURE;
  }
  while (fgets(line, sizeof line, stdin)) {
    size_t len = strcspn(line, "\n");
    int matches[2];
    int offset = 0;
    int flags = 0;
    line[len] = '\0';
    while (0 < pcre_exec(expr, 0, line, len, offset, flags, matches, 2)) {
      printf("%.*s%s%.*s%s",
        matches[0] - offset, line + offset,
        enter_reverse_mode,
        matches[1] - matches[0], line + matches[0],
        exit_reverse_mode);
      offset = matches[1];
      flags |= PCRE_NOTBOL;
    }
    printf("%s\n", line + offset);
  }
  return EXIT_SUCCESS;
}


