Introduction
============

Markup
------
*italic*, **bold**, `link <www.python.org>`_

all headings can be linked to elsewhere in the same file: Markup_

* This is a bulleted list.
* It has two items, the second
  item uses two lines. (note the indentation)

1. This is a numbered list.
2. It has two items too.

#. This continues the numbered list.

Display a code block::
    import slab
    print('Hello world!')

To highlight a note in a box:

.. note::  Use this for summaries or other important points to remember.

Or highlight a warning in a red box:

.. warning:: Use this for exception or other unexpected behavior of the code.

Add a footnote [#f1]_ .

Test-render your text `here. <http://rst.ninjs.org/?theme=nature>`_

As described in [Hofman98]_, you can cite papers. The references should be at the end of the document, but it may be more useful to directly link to the pubmed page: `Hofman (1998) <https://pubmed.ncbi.nlm.nih.gov/10196533/>`


.. rubric:: Footnotes
.. [#f1] if necessary.

.. rubric:: References
.. [Hofman98] Hofman (1998) Relearning sound localization with new ears. Nat Neurosci 1(5):417-21
