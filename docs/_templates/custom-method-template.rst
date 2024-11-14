{{ name | escape | underline }}

.. currentmodule:: {{ module }}

.. autofunction:: {{ objname }}

{% block parameters %}
{% if parameters %}
.. rubric:: {{ _('Parameters') }}

.. autosummary::
{% for param in parameters %}
   ~{{ param.name }}: {{ param.type }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block return %}
{% if return %}
.. rubric:: {{ _('Return Type') }}

.. autosummary::
   :toctree:
   ~{{ return.type }}
{% endif %}
{% endblock %}

{% block docstring %}
{% if docstring %}
.. rubric:: {{ _('Docstring') }}

{{ docstring | indent(3) }}
{% endif %}
{% endblock %}

{% block methods %}
{% if methods %}
.. rubric:: {{ _('Methods') }}

.. autosummary::
   :toctree:
{% for item in methods %}
   ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block attributes %}
{% if attributes %}
.. rubric:: {{ _('Attributes') }}

.. autosummary::
   :toctree:
{% for item in attributes %}
   ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

