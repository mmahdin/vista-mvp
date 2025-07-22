from django import forms


class RideBookingForm(forms.Form):
    ORIGIN = 'origin'
    DESTINATION = 'destination'

    ride_type = forms.ChoiceField(
        choices=[
            ('standard', 'Standard'),
            ('premium', 'Premium'),
            ('group', 'Group Ride'),
            ('cargo', 'Cargo')
        ],
        widget=forms.RadioSelect(attrs={'class': 'btn-check'})
    )
    notes = forms.CharField(
        widget=forms.Textarea(attrs={
            'rows': 3,
            'placeholder': 'Special instructions for driver...'
        }),
        required=False
    )
    payment_method = forms.ChoiceField(
        choices=[
            ('credit', 'Credit Card'),
            ('wallet', 'Wallet'),
            ('cash', 'Cash')
        ],
        widget=forms.RadioSelect(attrs={'class': 'btn-check'})
    )

    # Hidden fields for coordinates
    origin_lat = forms.FloatField(widget=forms.HiddenInput())
    origin_lng = forms.FloatField(widget=forms.HiddenInput())
    destination_lat = forms.FloatField(widget=forms.HiddenInput())
    destination_lng = forms.FloatField(widget=forms.HiddenInput())
