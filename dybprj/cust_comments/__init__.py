from django import forms
from django.utils.translation import ugettext_lazy as _
from django.contrib.comments.forms import CommentForm
from django.conf import settings

COMMENT_MAX_LENGTH = getattr(settings,'COMMENT_MAX_LENGTH', 3000)

class CustCommentForm(CommentForm):
    name          = forms.CharField(label=_("Name"), max_length=50, required=False )
    email         = forms.EmailField(label=_("Email address") , required=False)
    comment       = forms.CharField(label=_('Comment'), widget=forms.Textarea( attrs={'cols':'80' , 'class':'comment'} ), max_length=COMMENT_MAX_LENGTH)


def get_form():
    return CustCommentForm


