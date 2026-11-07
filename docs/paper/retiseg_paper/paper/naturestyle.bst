%% naturestyle.bst -- numbered references in the style of the Nature journals,
%% in order of first citation (for natbib with the "numbers" option).
%%
%%   article        Surname, A. B., Surname, C. & Surname, D. Title. J. Abbrev. 12, 34--56 (2020).
%%   inproceedings  Surname, A. et al. Title. In Proc. Booktitle 34--56 (Publisher, 2020).
%%   misc (arXiv)   Surname, A. et al. Title. Preprint at https://arxiv.org/abs/2401.00001 (2024).
%%
%% More than five authors: the first author followed by "et al.".
%% Journal names are printed as given in the journal field; refs.bib holds the
%% abbreviated names. The title links to https://doi.org/<doi> when a doi is given.
%% Written for the RetiSeg paper (2026); derived in structure from plain/unsrtnat.

ENTRY
  { address author booktitle chapter doi edition editor eprint howpublished
    institution journal key note number organization pages publisher school
    series title type url volume year }
  {}
  { label }

INTEGERS { nameptr namesleft numnames }
STRINGS { s t }

FUNCTION {not} { { #0 } { #1 } if$ }
FUNCTION {and} { 'skip$ { pop$ #0 } if$ }
FUNCTION {or} { { pop$ #1 } 'skip$ if$ }

FUNCTION {field.or.null}
{ duplicate$ empty$ { pop$ "" } 'skip$ if$ }

FUNCTION {emphasize}
{ duplicate$ empty$ { pop$ "" } { "\textit{" swap$ * "}" * } if$ }

FUNCTION {embolden}
{ duplicate$ empty$ { pop$ "" } { "\textbf{" swap$ * "}" * } if$ }

FUNCTION {space.if.nonempty}
{ duplicate$ empty$ 'skip$ { " " * } if$ }

% --- authors -----------------------------------------------------------
FUNCTION {format.names}
{ 's :=
  s num.names$ 'numnames :=
  numnames #5 >
    { s #1 "{vv~}{ll}{, jj}{, f.}" format.name$ " et~al." * }
    { #1 'nameptr :=
      numnames 'namesleft :=
      ""
      { namesleft #0 > }
      { s nameptr "{vv~}{ll}{, jj}{, f.}" format.name$ 't :=
        nameptr #1 >
          { namesleft #1 >
              { ", " * t * }
              { t "others" =
                  { " et~al." * }
                  { " \& " * t * }
                if$
              }
            if$
          }
          { pop$ t }
        if$
        nameptr #1 + 'nameptr :=
        namesleft #1 - 'namesleft :=
      }
      while$
    }
  if$
}

FUNCTION {format.authors}
{ author empty$
    { editor empty$ { "" } { editor format.names add.period$ } if$ }
    { author format.names add.period$ }
  if$
}

FUNCTION {format.editors.inline}
{ editor empty$
    { "" }
    { editor num.names$ #1 >
        { "(eds " editor format.names * ")" * }
        { "(ed. " editor format.names * ")" * }
      if$
    }
  if$
}

% --- title (linked to the DOI) -----------------------------------------
FUNCTION {format.title}
{ title empty$
    { "" }
    { doi empty$
        { title add.period$ }
        { "\href{https://doi.org/" doi * "}{" * title * "}" * add.period$ }
      if$
    }
  if$
}

FUNCTION {format.btitle.linked}
{ title empty$
    { "" }
    { doi empty$
        { title emphasize add.period$ }
        { "\href{https://doi.org/" doi * "}{" * title emphasize * "}" * add.period$ }
      if$
    }
  if$
}

% --- pages: single hyphen to en dash ------------------------------------
FUNCTION {n.dashify}
{ 't :=
  ""
  { t empty$ not }
  { t #1 #1 substring$ "-" =
      { t #1 #2 substring$ "--" = not
          { "--" * t #2 global.max$ substring$ 't := }
          { { t #1 #1 substring$ "-" = }
            { "-" * t #2 global.max$ substring$ 't := }
            while$
          }
        if$
      }
      { t #1 #1 substring$ * t #2 global.max$ substring$ 't := }
    if$
  }
  while$
}

FUNCTION {format.pages}
{ pages empty$ { "" } { pages n.dashify } if$ }

FUNCTION {format.year.paren}
{ year empty$ { "" } { "(" year * ")" * } if$ }

FUNCTION {format.url.text}
{ url empty$
    { eprint empty$
        { howpublished empty$ { "" } { howpublished } if$ }
        { "\url{https://arxiv.org/abs/" eprint * "}" * }
      if$
    }
    { "\url{" url * "}" * }
  if$
}

% --- \bibitem -----------------------------------------------------------
FUNCTION {output.bibitem}
{ newline$
  "\bibitem{" write$
  cite$ write$
  "}" write$
  newline$
}

FUNCTION {fin.entry}
{ write$ newline$ }

% --- entry types --------------------------------------------------------
% Author list. Title. Journal vol, pages (year).
FUNCTION {article}
{ output.bibitem
  format.authors space.if.nonempty
  format.title space.if.nonempty *
  journal emphasize *
  volume empty$
    { pages empty$
        { " " * format.year.paren * }
        { " " * format.pages * " " * format.year.paren * }
      if$
    }
    { " " * volume embolden *
      pages empty$
        { " " * format.year.paren * }
        { ", " * format.pages * " " * format.year.paren * }
      if$
    }
  if$
  "." *
  fin.entry
}

% Author list. Title. In Proc. Booktitle (eds ...) pages (Publisher, year).
FUNCTION {inproceedings}
{ output.bibitem
  format.authors space.if.nonempty
  format.title space.if.nonempty *
  "In " * booktitle emphasize *
  volume empty$ 'skip$ { series empty$ { " Vol.~" * volume * pages empty$ 'skip$ { "," * } if$ } 'skip$ if$ } if$
  editor empty$ 'skip$ { " " * format.editors.inline * } if$
  pages empty$ 'skip$ { " " * format.pages * } if$
  " (" *
  publisher empty$ 'skip$ { publisher * ", " * } if$
  year field.or.null * ")." *
  fin.entry
}

FUNCTION {incollection} { inproceedings }
FUNCTION {conference} { inproceedings }

% Author list. Title (Publisher, year).
FUNCTION {book}
{ output.bibitem
  format.authors space.if.nonempty
  title empty$ 'skip$
    { doi empty$
        { title emphasize * }
        { "\href{https://doi.org/" doi * "}{" * title emphasize * "}" * * }
      if$
    }
  if$
  " (" *
  publisher empty$ 'skip$ { publisher * ", " * } if$
  year field.or.null * ")." *
  fin.entry
}

% Author list. Title. Tech. Rep. number (Institution, year).
FUNCTION {techreport}
{ output.bibitem
  format.authors space.if.nonempty
  format.title space.if.nonempty *
  type empty$ { "Technical Report" } { type } if$ *
  number empty$ 'skip$ { " " * number * } if$
  " (" *
  institution empty$ 'skip$ { institution * ", " * } if$
  year field.or.null * ")." *
  fin.entry
}

% Author list. Title. Preprint at URL (year).   or   Author list. Title. Howpublished (year).
FUNCTION {misc}
{ output.bibitem
  format.authors space.if.nonempty
  format.title space.if.nonempty *
  eprint empty$ url empty$ and
    { howpublished empty$ 'skip$ { howpublished * " " * } if$ }
    { note empty$ { "Preprint at " } { note " " * } if$ * format.url.text * " " * }
  if$
  format.year.paren * "." *
  fin.entry
}

FUNCTION {unpublished} { misc }
FUNCTION {manual} { misc }
FUNCTION {phdthesis} { techreport }
FUNCTION {mastersthesis} { techreport }
FUNCTION {default.type} { misc }

% --- driver -------------------------------------------------------------
READ

STRINGS { longest.label }
INTEGERS { number.label longest.label.width }

FUNCTION {initialize.longest.label}
{ "" 'longest.label :=
  #1 'number.label :=
  #0 'longest.label.width :=
}

FUNCTION {longest.label.pass}
{ number.label int.to.str$ 'label :=
  number.label #1 + 'number.label :=
  label width$ longest.label.width >
    { label 'longest.label :=
      label width$ 'longest.label.width :=
    }
    'skip$
  if$
}

EXECUTE {initialize.longest.label}
ITERATE {longest.label.pass}

FUNCTION {begin.bib}
{ preamble$ empty$ 'skip$ { preamble$ write$ newline$ } if$
  "\providecommand{\url}[1]{\texttt{#1}}" write$ newline$
  "\providecommand{\href}[2]{#2}" write$ newline$
  "\begin{thebibliography}{" longest.label * "}" * write$ newline$
}

EXECUTE {begin.bib}
ITERATE {call.type$}

FUNCTION {end.bib}
{ newline$ "\end{thebibliography}" write$ newline$ }

EXECUTE {end.bib}
